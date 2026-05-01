// rife implemented with ncnn library

#include "rife.h"
//#include <iostream>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "benchmark.h"

#include "rife_preproc.comp.hex.h"
#include "rife_postproc.comp.hex.h"
#include "rife_v4_timestep.comp.hex.h"

#include "rife_ops.h"

DEFINE_LAYER_CREATOR(Warp)

RIFE::RIFE(int gpuid, float _flow_scale, int _num_threads, bool _rife_v2, bool _rife_v4, int _padding)
{
    vkdev = gpuid == -1 ? 0 : ncnn::get_gpu_device(gpuid);

    rife_preproc = 0;
    rife_postproc = 0;
    rife_v4_timestep = 0;
    rife_flow_scale_image = 0;
    rife_flow_resize_flow = 0;
    rife_flow_scale_vectors = 0;
    rife_v2_slice_flow = 0;
    use_flow_scale = std::abs(_flow_scale - 1.f) > 1e-6f;
    flow_scale = _flow_scale;
    flow_scale_inv = 1.f / _flow_scale;
    num_threads = _num_threads;
    rife_v2 = _rife_v2;
    rife_v4 = _rife_v4;
    padding = _padding;
}

RIFE::~RIFE()
{
    // cleanup preprocess and postprocess pipeline
    {
        delete rife_preproc;
        delete rife_postproc;
        delete rife_v4_timestep;
    }

    if (use_flow_scale)
    {
        rife_flow_scale_image->destroy_pipeline(flownet.opt);
        delete rife_flow_scale_image;

        rife_flow_resize_flow->destroy_pipeline(flownet.opt);
        delete rife_flow_resize_flow;

        rife_flow_scale_vectors->destroy_pipeline(flownet.opt);
        delete rife_flow_scale_vectors;
    }

    if (rife_v2)
    {
        rife_v2_slice_flow->destroy_pipeline(flownet.opt);
        delete rife_v2_slice_flow;
    }
}

#if _WIN32
static void load_param_model(ncnn::Net& net, const std::wstring& modeldir, const wchar_t* name)
{
    wchar_t parampath[256];
    wchar_t modelpath[256];
    swprintf(parampath, 256, L"%s/%s.param", modeldir.c_str(), name);
    swprintf(modelpath, 256, L"%s/%s.bin", modeldir.c_str(), name);

    {
        FILE* fp = _wfopen(parampath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", parampath);
        }

        net.load_param(fp);

        fclose(fp);
    }
    {
        FILE* fp = _wfopen(modelpath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", modelpath);
        }

        net.load_model(fp);

        fclose(fp);
    }
}
#else
static void load_param_model(ncnn::Net& net, const std::string& modeldir, const char* name)
{
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s/%s.param", modeldir.c_str(), name);
    sprintf(modelpath, "%s/%s.bin", modeldir.c_str(), name);

    net.load_param(parampath);
    net.load_model(modelpath);
}
#endif

#if _WIN32
int RIFE::load(const std::wstring& modeldir)
#else
int RIFE::load(const std::string& modeldir)
#endif
{
    ncnn::Option opt;
    opt.num_threads = num_threads;
    opt.use_vulkan_compute = vkdev ? true : false;
    opt.use_fp16_packed = vkdev ? true : false;
    opt.use_fp16_storage = vkdev ? true : false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;

    flownet.opt = opt;
    contextnet.opt = opt;
    fusionnet.opt = opt;

    flownet.set_vulkan_device(vkdev);
    contextnet.set_vulkan_device(vkdev);
    fusionnet.set_vulkan_device(vkdev);

    flownet.register_custom_layer("rife.Warp", Warp_layer_creator);
    contextnet.register_custom_layer("rife.Warp", Warp_layer_creator);
    fusionnet.register_custom_layer("rife.Warp", Warp_layer_creator);

#if _WIN32
    load_param_model(flownet, modeldir, L"flownet");
    if (!rife_v4)
    {
        load_param_model(contextnet, modeldir, L"contextnet");
        load_param_model(fusionnet, modeldir, L"fusionnet");
    }
#else
    load_param_model(flownet, modeldir, "flownet");
    if (!rife_v4)
    {
        load_param_model(contextnet, modeldir, "contextnet");
        load_param_model(fusionnet, modeldir, "fusionnet");
    }
#endif

    // initialize preprocess and postprocess pipeline
    if (vkdev)
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        {
            std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                    compile_spirv_module(rife_preproc_comp_data, sizeof(rife_preproc_comp_data), opt, spirv);
            }

            rife_preproc = new ncnn::Pipeline(vkdev);
            rife_preproc->set_optimal_local_size_xyz(8, 8, 3);
            rife_preproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                    compile_spirv_module(rife_postproc_comp_data, sizeof(rife_postproc_comp_data), opt, spirv);
            }

            rife_postproc = new ncnn::Pipeline(vkdev);
            rife_postproc->set_optimal_local_size_xyz(8, 8, 3);
            rife_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    if (use_flow_scale)
    {
        {
            rife_flow_scale_image = ncnn::create_layer("Interp");
            rife_flow_scale_image->vkdev = vkdev;

            ncnn::ParamDict pd;
            pd.set(0, 2);// bilinear
            pd.set(1, flow_scale);
            pd.set(2, flow_scale);
            rife_flow_scale_image->load_param(pd);

            rife_flow_scale_image->create_pipeline(opt);
        }
        {
            rife_flow_resize_flow = ncnn::create_layer("Interp");
            rife_flow_resize_flow->vkdev = vkdev;

            ncnn::ParamDict pd;
            pd.set(0, 2);// bilinear
            pd.set(1, flow_scale_inv);
            pd.set(2, flow_scale_inv);
            rife_flow_resize_flow->load_param(pd);

            rife_flow_resize_flow->create_pipeline(opt);
        }
        {
            rife_flow_scale_vectors = ncnn::create_layer("BinaryOp");
            rife_flow_scale_vectors->vkdev = vkdev;

            ncnn::ParamDict pd;
            pd.set(0, 2);// mul
            pd.set(1, 1);// with_scalar
            pd.set(2, flow_scale_inv);// b
            rife_flow_scale_vectors->load_param(pd);

            rife_flow_scale_vectors->create_pipeline(opt);
        }
    }

    if (rife_v2)
    {
        {
            rife_v2_slice_flow = ncnn::create_layer("Slice");
            rife_v2_slice_flow->vkdev = vkdev;

            ncnn::Mat slice_points(2);
            slice_points.fill<int>(-233);

            ncnn::ParamDict pd;
            pd.set(0, slice_points);
            pd.set(1, 0);// axis

            rife_v2_slice_flow->load_param(pd);

            rife_v2_slice_flow->create_pipeline(opt);
        }
    }

    if (rife_v4)
    {
        if (vkdev)
        {
            std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    compile_spirv_module(rife_v4_timestep_comp_data, sizeof(rife_v4_timestep_comp_data), opt, spirv);
                }
            }

            std::vector<ncnn::vk_specialization_type> specializations;

            rife_v4_timestep = new ncnn::Pipeline(vkdev);
            rife_v4_timestep->set_optimal_local_size_xyz(8, 8, 1);
            rife_v4_timestep->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    return 0;
}

static float convert_fp16_to_float32(const uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t mantissa = value & 0x03ffu;
    uint32_t bits{};

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            bits = sign;
        }
        else
        {
            auto normalized_mantissa = mantissa;
            int adjusted_exponent = -14;
            while ((normalized_mantissa & 0x0400u) == 0)
            {
                normalized_mantissa <<= 1;
                adjusted_exponent -= 1;
            }

            normalized_mantissa &= 0x03ffu;
            bits = sign | (static_cast<uint32_t>(adjusted_exponent + 127) << 23) | (normalized_mantissa << 13);
        }
    }
    else if (exponent == 0x1fu)
    {
        bits = sign | 0x7f800000u | (mantissa << 13);
    }
    else
    {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static float read_ncnn_scalar(const unsigned char* const data, const size_t index, const int scalar_size)
{
    if (scalar_size == static_cast<int>(sizeof(float)))
    {
        float value;
        std::memcpy(&value, data + index * sizeof(float), sizeof(value));
        return value;
    }

    uint16_t value;
    std::memcpy(&value, data + index * sizeof(uint16_t), sizeof(value));
    return convert_fp16_to_float32(value);
}

static float sample_bilinear_channel(const float* const data, const int w, const int h, float x, float y)
{
    x = std::max(0.0f, std::min(x, static_cast<float>(w - 1)));
    y = std::max(0.0f, std::min(y, static_cast<float>(h - 1)));

    const auto x0 = static_cast<int>(std::floor(x));
    const auto y0 = static_cast<int>(std::floor(y));
    const auto x1 = std::min(x0 + 1, w - 1);
    const auto y1 = std::min(y0 + 1, h - 1);

    const auto alpha = x - x0;
    const auto beta = y - y0;
    const auto row0 = y0 * w;
    const auto row1 = y1 * w;
    const auto v00 = data[row0 + x0];
    const auto v01 = data[row0 + x1];
    const auto v10 = data[row1 + x0];
    const auto v11 = data[row1 + x1];
    const auto top = v00 * (1.0f - alpha) + v01 * alpha;
    const auto bottom = v10 * (1.0f - alpha) + v11 * alpha;

    return top * (1.0f - beta) + bottom * beta;
}

int RIFE::process_flow_native(const float* src0R, const float* src0G, const float* src0B,
                              const float* src1R, const float* src1G, const float* src1B,
                              std::vector<float>& flow_out, int& flow_w, int& flow_h,
                              const int w, const int h, const ptrdiff_t stride) const
{
    flow_out.clear();
    flow_w = 0;
    flow_h = 0;

    if (rife_v4)
        return -1;

    const int channels = 3;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    const auto reclaim_allocators = [&]() {
        vkdev->reclaim_blob_allocator(blob_vkallocator);
        vkdev->reclaim_staging_allocator(staging_vkallocator);
    };

    ncnn::Option opt = flownet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    const auto w_padded = (w + padding - 1) / padding * padding;
    const auto h_padded = (h + padding - 1) / padding * padding;
    const auto in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::Mat in0;
    ncnn::Mat in1;
    in0.create(w, h, channels, sizeof(float), 1);
    in1.create(w, h, channels, sizeof(float), 1);
    auto in0_r = in0.channel(0);
    auto in0_g = in0.channel(1);
    auto in0_b = in0.channel(2);
    auto in1_r = in1.channel(0);
    auto in1_g = in1.channel(1);
    auto in1_b = in1.channel(2);
    for (auto y = 0; y < h; y++) {
        for (auto x = 0; x < w; x++) {
            in0_r[w * y + x] = src0R[stride * y + x] * 255.0f;
            in0_g[w * y + x] = src0G[stride * y + x] * 255.0f;
            in0_b[w * y + x] = src0B[stride * y + x] * 255.0f;
            in1_r[w * y + x] = src1R[stride * y + x] * 255.0f;
            in1_g[w * y + x] = src1G[stride * y + x] * 255.0f;
            in1_b[w * y + x] = src1B[stride * y + x] * 255.0f;
        }
    }

    ncnn::VkCompute cmd(vkdev);

    ncnn::VkMat in0_gpu;
    ncnn::VkMat in1_gpu;
    cmd.record_clone(in0, in0_gpu, opt);
    cmd.record_clone(in1, in1_gpu, opt);

    ncnn::VkMat in0_gpu_padded;
    ncnn::VkMat in1_gpu_padded;
    {
        in0_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = in0_gpu;
        bindings[1] = in0_gpu_padded;

        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].i = in0_gpu.w;
        constants[1].i = in0_gpu.h;
        constants[2].i = in0_gpu.cstep;
        constants[3].i = in0_gpu_padded.w;
        constants[4].i = in0_gpu_padded.h;
        constants[5].i = in0_gpu_padded.cstep;

        cmd.record_pipeline(rife_preproc, bindings, constants, in0_gpu_padded);
    }
    {
        in1_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = in1_gpu;
        bindings[1] = in1_gpu_padded;

        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].i = in1_gpu.w;
        constants[1].i = in1_gpu.h;
        constants[2].i = in1_gpu.cstep;
        constants[3].i = in1_gpu_padded.w;
        constants[4].i = in1_gpu_padded.h;
        constants[5].i = in1_gpu_padded.cstep;

        cmd.record_pipeline(rife_preproc, bindings, constants, in1_gpu_padded);
    }

    ncnn::VkMat flow;
    {
        ncnn::Extractor ex = flownet.create_extractor();
        ex.set_blob_vkallocator(blob_vkallocator);
        ex.set_workspace_vkallocator(blob_vkallocator);
        ex.set_staging_vkallocator(staging_vkallocator);

        if (use_flow_scale)
        {
            ncnn::VkMat in0_gpu_padded_downscaled;
            ncnn::VkMat in1_gpu_padded_downscaled;
            rife_flow_scale_image->forward(in0_gpu_padded, in0_gpu_padded_downscaled, cmd, opt);
            rife_flow_scale_image->forward(in1_gpu_padded, in1_gpu_padded_downscaled, cmd, opt);

            ex.input("input0", in0_gpu_padded_downscaled);
            ex.input("input1", in1_gpu_padded_downscaled);

            ncnn::VkMat flow_downscaled;
            ex.extract("flow", flow_downscaled, cmd);

            ncnn::VkMat flow_half;
            rife_flow_resize_flow->forward(flow_downscaled, flow_half, cmd, opt);
            rife_flow_scale_vectors->forward(flow_half, flow, cmd, opt);
        }
        else
        {
            ex.input("input0", in0_gpu_padded);
            ex.input("input1", in1_gpu_padded);
            ex.extract("flow", flow, cmd);
        }
    }

    ncnn::Mat flow_cpu;
    cmd.record_clone(flow, flow_cpu, opt);
    cmd.submit_and_wait();

    ncnn::Mat flow_cpu_unpacked;
    const auto scalar_size = flow_cpu.elempack > 0 ? static_cast<int>(flow_cpu.elemsize / flow_cpu.elempack) : 0;
    if (flow_cpu.elempack < 1 || (scalar_size != static_cast<int>(sizeof(float)) && scalar_size != static_cast<int>(sizeof(uint16_t))))
    {
        reclaim_allocators();
        return -1;
    }

    if (flow_cpu.elempack != 1 || scalar_size != static_cast<int>(sizeof(float)))
    {
        const int ep = flow_cpu.elempack;
        const int actual_c = flow_cpu.c * ep;
        flow_cpu_unpacked.create(flow_cpu.w, flow_cpu.h, actual_c, sizeof(float));
        const auto pixel_count = static_cast<size_t>(flow_cpu.w) * flow_cpu.h;
        for (int cg = 0; cg < flow_cpu.c; cg++)
        {
            const auto* packed = static_cast<const unsigned char*>(flow_cpu.channel(cg));
            for (int ep_idx = 0; ep_idx < ep; ep_idx++)
            {
                auto* dst = static_cast<float*>(flow_cpu_unpacked.channel(cg * ep + ep_idx));
                for (size_t i = 0; i < pixel_count; i++)
                {
                    dst[i] = read_ncnn_scalar(packed, i * ep + ep_idx, scalar_size);
                }
            }
        }
    }
    else
    {
        flow_cpu_unpacked = flow_cpu;
    }

    if (flow_cpu_unpacked.c < 4)
    {
        reclaim_allocators();
        return -1;
    }

    flow_w = flow_cpu_unpacked.w;
    flow_h = flow_cpu_unpacked.h;
    const auto plane_size = static_cast<size_t>(flow_w) * flow_h;
    flow_out.resize(plane_size * 4);
    for (auto c = 0; c < 4; c++)
    {
        const auto src = static_cast<const float*>(flow_cpu_unpacked.channel(c));
        auto* dst = flow_out.data() + static_cast<size_t>(c) * plane_size;
        std::memcpy(dst, src, plane_size * sizeof(float));
    }

    reclaim_allocators();

    return 0;
}

int RIFE::process_flow(const float* src0R, const float* src0G, const float* src0B,
                       const float* src1R, const float* src1G, const float* src1B,
                       float* flow_out, const int w, const int h, const ptrdiff_t stride) const
{
    std::vector<float> flow_native;
    int flow_w{};
    int flow_h{};
    if (process_flow_native(src0R, src0G, src0B, src1R, src1G, src1B, flow_native, flow_w, flow_h, w, h, stride) != 0)
        return -1;

    const auto plane_size = static_cast<size_t>(flow_w) * flow_h;
    const auto out_w = flow_w * 2;
    const auto out_h = flow_h * 2;
    for (auto c = 0; c < 4; c++)
    {
        const auto* src = flow_native.data() + static_cast<size_t>(c) * plane_size;
        auto* dst = flow_out + static_cast<size_t>(c) * w * h;
        for (auto y = 0; y < h; y++)
        {
            const auto sample_y = (y + 0.5f) * flow_h / static_cast<float>(out_h) - 0.5f;
            for (auto x = 0; x < w; x++)
            {
                const auto sample_x = (x + 0.5f) * flow_w / static_cast<float>(out_w) - 0.5f;
                dst[w * y + x] = 2.0f * sample_bilinear_channel(src, flow_w, flow_h, sample_x, sample_y);
            }
        }
    }

    return 0;
}

int RIFE::process(const float* src0R, const float* src0G, const float* src0B,
                  const float* src1R, const float* src1G, const float* src1B,
                  float* dstR, float* dstG, float* dstB,
                  const int w, const int h, const ptrdiff_t stride, const float timestep) const
{
    if (rife_v4)
        return process_v4(src0R, src0G, src0B, src1R, src1G, src1B, dstR, dstG, dstB, w, h, stride, timestep);

    const int channels = 3;//in0image.elempack;

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = flownet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // pad to 32n
    int w_padded = (w + padding - 1) / padding * padding;
    int h_padded = (h + padding - 1) / padding * padding;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::Mat in0;
    ncnn::Mat in1;
    in0.create(w, h, channels, sizeof(float), 1);
    in1.create(w, h, channels, sizeof(float), 1);
    float* in0R{ in0.channel(0) };
    float* in0G{ in0.channel(1) };
    float* in0B{ in0.channel(2) };
    float* in1R{ in1.channel(0) };
    float* in1G{ in1.channel(1) };
    float* in1B{ in1.channel(2) };
    for (auto y{ 0 }; y < h; y++) {
        for (auto x{ 0 }; x < w; x++) {
            in0R[w * y + x] = src0R[stride * y + x] * 255.0f;
            in0G[w * y + x] = src0G[stride * y + x] * 255.0f;
            in0B[w * y + x] = src0B[stride * y + x] * 255.0f;
            in1R[w * y + x] = src1R[stride * y + x] * 255.0f;
            in1G[w * y + x] = src1G[stride * y + x] * 255.0f;
            in1B[w * y + x] = src1B[stride * y + x] * 255.0f;
        }
    }

    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat in0_gpu;
    ncnn::VkMat in1_gpu;
    {
        cmd.record_clone(in0, in0_gpu, opt);
        cmd.record_clone(in1, in1_gpu, opt);
    }

    ncnn::VkMat out_gpu;

    // preproc
    ncnn::VkMat in0_gpu_padded;
    ncnn::VkMat in1_gpu_padded;
    {
        in0_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = in0_gpu;
        bindings[1] = in0_gpu_padded;

        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].i = in0_gpu.w;
        constants[1].i = in0_gpu.h;
        constants[2].i = in0_gpu.cstep;
        constants[3].i = in0_gpu_padded.w;
        constants[4].i = in0_gpu_padded.h;
        constants[5].i = in0_gpu_padded.cstep;

        cmd.record_pipeline(rife_preproc, bindings, constants, in0_gpu_padded);
    }
    {
        in1_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = in1_gpu;
        bindings[1] = in1_gpu_padded;

        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].i = in1_gpu.w;
        constants[1].i = in1_gpu.h;
        constants[2].i = in1_gpu.cstep;
        constants[3].i = in1_gpu_padded.w;
        constants[4].i = in1_gpu_padded.h;
        constants[5].i = in1_gpu_padded.cstep;

        cmd.record_pipeline(rife_preproc, bindings, constants, in1_gpu_padded);
    }

    // flownet
    ncnn::VkMat flow;
    ncnn::VkMat flow0;
    ncnn::VkMat flow1;
    {
        ncnn::Extractor ex = flownet.create_extractor();
        ex.set_blob_vkallocator(blob_vkallocator);
        ex.set_workspace_vkallocator(blob_vkallocator);
        ex.set_staging_vkallocator(staging_vkallocator);

        if (use_flow_scale)
        {
            ncnn::VkMat in0_gpu_padded_downscaled;
            ncnn::VkMat in1_gpu_padded_downscaled;
            rife_flow_scale_image->forward(in0_gpu_padded, in0_gpu_padded_downscaled, cmd, opt);
            rife_flow_scale_image->forward(in1_gpu_padded, in1_gpu_padded_downscaled, cmd, opt);

            ex.input("input0", in0_gpu_padded_downscaled);
            ex.input("input1", in1_gpu_padded_downscaled);

            ncnn::VkMat flow_downscaled;
            ex.extract("flow", flow_downscaled, cmd);

            ncnn::VkMat flow_half;
            rife_flow_resize_flow->forward(flow_downscaled, flow_half, cmd, opt);

            rife_flow_scale_vectors->forward(flow_half, flow, cmd, opt);
        }
        else
        {
            ex.input("input0", in0_gpu_padded);
            ex.input("input1", in1_gpu_padded);
            ex.extract("flow", flow, cmd);
        }
    }

    if (rife_v2)
    {
        std::vector<ncnn::VkMat> inputs(1);
        inputs[0] = flow;
        std::vector<ncnn::VkMat> outputs(2);
        rife_v2_slice_flow->forward(inputs, outputs, cmd, opt);
        flow0 = outputs[0];
        flow1 = outputs[1];
    }

    // contextnet
    ncnn::VkMat ctx0[4];
    ncnn::VkMat ctx1[4];
    {
        ncnn::Extractor ex = contextnet.create_extractor();
        ex.set_blob_vkallocator(blob_vkallocator);
        ex.set_workspace_vkallocator(blob_vkallocator);
        ex.set_staging_vkallocator(staging_vkallocator);

        ex.input("input.1", in0_gpu_padded);
        if (rife_v2)
        {
            ex.input("flow.0", flow0);
        }
        else
        {
            ex.input("flow.0", flow);
        }
        ex.extract("f1", ctx0[0], cmd);
        ex.extract("f2", ctx0[1], cmd);
        ex.extract("f3", ctx0[2], cmd);
        ex.extract("f4", ctx0[3], cmd);
    }
    {
        ncnn::Extractor ex = contextnet.create_extractor();
        ex.set_blob_vkallocator(blob_vkallocator);
        ex.set_workspace_vkallocator(blob_vkallocator);
        ex.set_staging_vkallocator(staging_vkallocator);

        ex.input("input.1", in1_gpu_padded);
        if (rife_v2)
        {
            ex.input("flow.0", flow1);
        }
        else
        {
            ex.input("flow.1", flow);
        }
        ex.extract("f1", ctx1[0], cmd);
        ex.extract("f2", ctx1[1], cmd);
        ex.extract("f3", ctx1[2], cmd);
        ex.extract("f4", ctx1[3], cmd);
    }

    // fusionnet
    ncnn::VkMat out_gpu_padded;
    {
        ncnn::Extractor ex = fusionnet.create_extractor();
        ex.set_blob_vkallocator(blob_vkallocator);
        ex.set_workspace_vkallocator(blob_vkallocator);
        ex.set_staging_vkallocator(staging_vkallocator);

        ex.input("img0", in0_gpu_padded);
        ex.input("img1", in1_gpu_padded);
        ex.input("flow", flow);
        ex.input("3", ctx0[0]);
        ex.input("4", ctx0[1]);
        ex.input("5", ctx0[2]);
        ex.input("6", ctx0[3]);
        ex.input("7", ctx1[0]);
        ex.input("8", ctx1[1]);
        ex.input("9", ctx1[2]);
        ex.input("10", ctx1[3]);

        in0_gpu.release();
        in1_gpu.release();
        ctx0[0].release();
        ctx0[1].release();
        ctx0[2].release();
        ctx0[3].release();
        ctx1[0].release();
        ctx1[1].release();
        ctx1[2].release();
        ctx1[3].release();
        flow.release();

        ex.extract("output", out_gpu_padded, cmd);
    }

    out_gpu.create(w, h, channels, sizeof(float), 1, blob_vkallocator);

    // postproc
    {
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = out_gpu_padded;
        bindings[1] = out_gpu;

        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].i = out_gpu_padded.w;
        constants[1].i = out_gpu_padded.h;
        constants[2].i = out_gpu_padded.cstep;
        constants[3].i = out_gpu.w;
        constants[4].i = out_gpu.h;
        constants[5].i = out_gpu.cstep;

        cmd.record_pipeline(rife_postproc, bindings, constants, out_gpu);
    }

    // download
    {
        ncnn::Mat out;

        cmd.record_clone(out_gpu, out, opt);

        cmd.submit_and_wait();

        const float* outR{ out.channel(0) };
        const float* outG{ out.channel(1) };
        const float* outB{ out.channel(2) };
        for (auto y{ 0 }; y < h; y++) {
            for (auto x{ 0 }; x < w; x++) {
                dstR[stride * y + x] = outR[w * y + x] * (1 / 255.0f);
                dstG[stride * y + x] = outG[w * y + x] * (1 / 255.0f);
                dstB[stride * y + x] = outB[w * y + x] * (1 / 255.0f);
            }
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int RIFE::process_v4(const float* src0R, const float* src0G, const float* src0B,
                     const float* src1R, const float* src1G, const float* src1B,
                     float* dstR, float* dstG, float* dstB,
                     const int w, const int h, const ptrdiff_t stride, const float timestep) const
{
    const int channels = 3;//in0image.elempack;

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = flownet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;
    // padding, the default is 32, but newer rife models require 64
    // std::cout << "padding: " << padding << std::endl;
    int w_padded = (w + padding - 1) / padding * padding;
    int h_padded = (h + padding - 1) / padding * padding;
    

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::Mat in0;
    ncnn::Mat in1;
    in0.create(w, h, channels, sizeof(float), 1);
    in1.create(w, h, channels, sizeof(float), 1);
    float* in0R{ in0.channel(0) };
    float* in0G{ in0.channel(1) };
    float* in0B{ in0.channel(2) };
    float* in1R{ in1.channel(0) };
    float* in1G{ in1.channel(1) };
    float* in1B{ in1.channel(2) };
    for (auto y{ 0 }; y < h; y++) {
        for (auto x{ 0 }; x < w; x++) {
            in0R[w * y + x] = src0R[stride * y + x] * 255.0f;
            in0G[w * y + x] = src0G[stride * y + x] * 255.0f;
            in0B[w * y + x] = src0B[stride * y + x] * 255.0f;
            in1R[w * y + x] = src1R[stride * y + x] * 255.0f;
            in1G[w * y + x] = src1G[stride * y + x] * 255.0f;
            in1B[w * y + x] = src1B[stride * y + x] * 255.0f;
        }
    }

    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat in0_gpu;
    ncnn::VkMat in1_gpu;
    {
        cmd.record_clone(in0, in0_gpu, opt);
        cmd.record_clone(in1, in1_gpu, opt);
    }

    ncnn::VkMat out_gpu;

    {
        // preproc
        ncnn::VkMat in0_gpu_padded;
        ncnn::VkMat in1_gpu_padded;
        ncnn::VkMat timestep_gpu_padded;
        {
            in0_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in0_gpu;
            bindings[1] = in0_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in0_gpu.w;
            constants[1].i = in0_gpu.h;
            constants[2].i = in0_gpu.cstep;
            constants[3].i = in0_gpu_padded.w;
            constants[4].i = in0_gpu_padded.h;
            constants[5].i = in0_gpu_padded.cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in0_gpu_padded);
        }
        {
            in1_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in1_gpu;
            bindings[1] = in1_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in1_gpu.w;
            constants[1].i = in1_gpu.h;
            constants[2].i = in1_gpu.cstep;
            constants[3].i = in1_gpu_padded.w;
            constants[4].i = in1_gpu_padded.h;
            constants[5].i = in1_gpu_padded.cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in1_gpu_padded);
        }
        {
            timestep_gpu_padded.create(w_padded, h_padded, 1, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(1);
            bindings[0] = timestep_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].i = timestep_gpu_padded.w;
            constants[1].i = timestep_gpu_padded.h;
            constants[2].i = timestep_gpu_padded.cstep;
            constants[3].f = timestep;

            cmd.record_pipeline(rife_v4_timestep, bindings, constants, timestep_gpu_padded);
        }

        // flownet
        ncnn::VkMat out_gpu_padded;
        {
            ncnn::Extractor ex = flownet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("in0", in0_gpu_padded);
            ex.input("in1", in1_gpu_padded);
            ex.input("in2", timestep_gpu_padded);
            ex.extract("out0", out_gpu_padded, cmd);
        }

        out_gpu.create(w, h, channels, sizeof(float), 1, blob_vkallocator);

        // postproc
        {
            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = out_gpu_padded;
            bindings[1] = out_gpu;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = out_gpu_padded.w;
            constants[1].i = out_gpu_padded.h;
            constants[2].i = out_gpu_padded.cstep;
            constants[3].i = out_gpu.w;
            constants[4].i = out_gpu.h;
            constants[5].i = out_gpu.cstep;

            cmd.record_pipeline(rife_postproc, bindings, constants, out_gpu);
        }
    }

    // download
    {
        ncnn::Mat out;

        cmd.record_clone(out_gpu, out, opt);

        cmd.submit_and_wait();

        const float* outR{ out.channel(0) };
        const float* outG{ out.channel(1) };
        const float* outB{ out.channel(2) };
        for (auto y{ 0 }; y < h; y++) {
            for (auto x{ 0 }; x < w; x++) {
                dstR[stride * y + x] = outR[w * y + x] * (1 / 255.0f);
                dstG[stride * y + x] = outG[w * y + x] * (1 / 255.0f);
                dstB[stride * y + x] = outB[w * y + x] * (1 / 255.0f);
            }
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

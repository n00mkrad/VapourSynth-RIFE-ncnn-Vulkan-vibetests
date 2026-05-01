// rife implemented with ncnn library

#include "rife.h"
//#include <iostream>

#include <array>
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

RIFE::RIFE(int gpuid, float _flow_scale, int _num_threads, bool _rife_v2, bool _rife_v4, int _padding, FlowResizeMode _flow_resize_mode)
{
    vkdev = gpuid == -1 ? 0 : ncnn::get_gpu_device(gpuid);

    rife_preproc = 0;
    rife_postproc = 0;
    rife_v4_timestep = 0;
    rife_flow_scale_image = 0;
    rife_flow_resize_flow = 0;
    rife_flow_scale_vectors = 0;
    rife_flow_resize_output = 0;
    rife_flow_double_vectors = 0;
    rife_v2_slice_flow = 0;
    use_flow_scale = std::abs(_flow_scale - 1.f) > 1e-6f;
    flow_scale = _flow_scale;
    flow_scale_inv = 1.f / _flow_scale;
    flow_resize_mode = _flow_resize_mode;
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

    if (rife_flow_resize_output)
    {
        rife_flow_resize_output->destroy_pipeline(flownet.opt);
        delete rife_flow_resize_output;
    }

    if (rife_flow_double_vectors)
    {
        rife_flow_double_vectors->destroy_pipeline(flownet.opt);
        delete rife_flow_double_vectors;
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

    if (vkdev)
    {
        rife_flow_resize_output = ncnn::create_layer("Interp");
        rife_flow_resize_output->vkdev = vkdev;
        {
            ncnn::ParamDict pd;
            pd.set(0, 2);// bilinear
            pd.set(1, 2.f);
            pd.set(2, 2.f);
            rife_flow_resize_output->load_param(pd);
        }
        rife_flow_resize_output->create_pipeline(opt);

        rife_flow_double_vectors = ncnn::create_layer("BinaryOp");
        rife_flow_double_vectors->vkdev = vkdev;
        {
            ncnn::ParamDict pd;
            pd.set(0, 2);// mul
            pd.set(1, 1);// with_scalar
            pd.set(2, 2.f);
            rife_flow_double_vectors->load_param(pd);
        }
        rife_flow_double_vectors->create_pipeline(opt);
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

struct BilinearAxisEntry {
    int index0;
    int index1;
    float alpha;
};

static void build_bilinear_axis_table(const int src_size, const int virtual_out_size, const int dst_size,
                                      std::vector<BilinearAxisEntry>& table) {
    table.resize(dst_size);
    const auto scale = src_size / static_cast<float>(virtual_out_size);
    for (int i = 0; i < dst_size; i++)
    {
        auto sample = (i + 0.5f) * scale - 0.5f;
        sample = std::max(0.0f, std::min(sample, static_cast<float>(src_size - 1)));
        const auto i0 = static_cast<int>(std::floor(sample));
        const auto i1 = std::min(i0 + 1, src_size - 1);
        table[i] = { i0, i1, sample - i0 };
    }
}

static int unpack_flow_channels(const ncnn::Mat& flow_cpu, ncnn::Mat& flow_cpu_unpacked)
{
    const auto scalar_size = flow_cpu.elempack > 0 ? static_cast<int>(flow_cpu.elemsize / flow_cpu.elempack) : 0;
    if (flow_cpu.elempack < 1 || (scalar_size != static_cast<int>(sizeof(float)) && scalar_size != static_cast<int>(sizeof(uint16_t))))
        return -1;

    if (flow_cpu.elempack == 1 && scalar_size == static_cast<int>(sizeof(float)))
    {
        flow_cpu_unpacked = flow_cpu;
        return 0;
    }

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
                dst[i] = read_ncnn_scalar(packed, i * ep + ep_idx, scalar_size);
        }
    }

    return 0;
}

static int copy_flow_output_direct(const ncnn::Mat& flow_cpu_unpacked, float* flow_out, const int w, const int h)
{
    if (flow_cpu_unpacked.c < 4 || flow_cpu_unpacked.w < w || flow_cpu_unpacked.h < h)
        return -1;

    for (int c = 0; c < 4; c++)
    {
        const auto* src = static_cast<const float*>(flow_cpu_unpacked.channel(c));
        auto* dst = flow_out + static_cast<size_t>(c) * w * h;
        for (int y = 0; y < h; y++)
            std::memcpy(dst + static_cast<size_t>(y) * w, src + static_cast<size_t>(y) * flow_cpu_unpacked.w, static_cast<size_t>(w) * sizeof(float));
    }

    return 0;
}

static int copy_flow_output_resized_cpu(const ncnn::Mat& flow_cpu_unpacked, float* flow_out, const int w, const int h)
{
    if (flow_cpu_unpacked.c < 4)
        return -1;

    const auto flow_w = flow_cpu_unpacked.w;
    const auto flow_h = flow_cpu_unpacked.h;
    const auto out_w = flow_w * 2;
    const auto out_h = flow_h * 2;

    struct AxisCache {
        int srcW{};
        int srcH{};
        int virtualOutW{};
        int virtualOutH{};
        int dstW{};
        int dstH{};
        std::vector<BilinearAxisEntry> xTable;
        std::vector<BilinearAxisEntry> yTable;
    };
    static thread_local AxisCache axisCache;
    if (axisCache.srcW != flow_w || axisCache.srcH != flow_h ||
        axisCache.virtualOutW != out_w || axisCache.virtualOutH != out_h ||
        axisCache.dstW != w || axisCache.dstH != h)
    {
        axisCache.srcW = flow_w;
        axisCache.srcH = flow_h;
        axisCache.virtualOutW = out_w;
        axisCache.virtualOutH = out_h;
        axisCache.dstW = w;
        axisCache.dstH = h;
        build_bilinear_axis_table(flow_w, out_w, w, axisCache.xTable);
        build_bilinear_axis_table(flow_h, out_h, h, axisCache.yTable);
    }

    for (int c = 0; c < 4; c++)
    {
        const auto* src = static_cast<const float*>(flow_cpu_unpacked.channel(c));
        auto* dst = flow_out + static_cast<size_t>(c) * w * h;
        for (int y = 0; y < h; y++)
        {
            const auto& yEntry = axisCache.yTable[y];
            const auto row0 = static_cast<size_t>(yEntry.index0) * flow_w;
            const auto row1 = static_cast<size_t>(yEntry.index1) * flow_w;
            for (int x = 0; x < w; x++)
            {
                const auto& xEntry = axisCache.xTable[x];
                const auto top = src[row0 + xEntry.index0] * (1.0f - xEntry.alpha) + src[row0 + xEntry.index1] * xEntry.alpha;
                const auto bottom = src[row1 + xEntry.index0] * (1.0f - xEntry.alpha) + src[row1 + xEntry.index1] * xEntry.alpha;
                dst[static_cast<size_t>(y) * w + x] = 2.0f * (top * (1.0f - yEntry.alpha) + bottom * yEntry.alpha);
            }
        }
    }

    return 0;
}

static bool extract_v4_flow_blob(ncnn::Extractor& ex, ncnn::VkCompute& cmd, ncnn::VkMat& flow)
{
    static constexpr std::array<const char*, 8> flow_blob_names{
        "/Add_4_output_0",
        "/Add_3_output_0",
        "Add_4_output_0",
        "Add_3_output_0",
        "/Add_2_output_0",
        "Add_2_output_0",
        "flow",
        "/flow"
    };

    for (const auto* const blob_name : flow_blob_names)
    {
        if (ex.extract(blob_name, flow, cmd) == 0 && !flow.empty())
            return true;
    }

    return false;
}

int RIFE::process_flow(const float* src0R, const float* src0G, const float* src0B,
                       const float* src1R, const float* src1G, const float* src1B,
                       float* flow_out, const int w, const int h, const ptrdiff_t stride) const
{
    const int channels = 3;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

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

        if (rife_v4)
        {
            ncnn::VkMat timestep_gpu_padded;
            timestep_gpu_padded.create(w_padded, h_padded, 1, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(1);
            bindings[0] = timestep_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].i = timestep_gpu_padded.w;
            constants[1].i = timestep_gpu_padded.h;
            constants[2].i = timestep_gpu_padded.cstep;
            constants[3].f = 0.5f;

            cmd.record_pipeline(rife_v4_timestep, bindings, constants, timestep_gpu_padded);

            ex.input("in0", in0_gpu_padded);
            ex.input("in1", in1_gpu_padded);
            ex.input("in2", timestep_gpu_padded);

            if (!extract_v4_flow_blob(ex, cmd, flow))
            {
                vkdev->reclaim_blob_allocator(blob_vkallocator);
                vkdev->reclaim_staging_allocator(staging_vkallocator);
                return -1;
            }
        }
        else if (use_flow_scale)
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
    bool used_gpu_resize{};
    const bool flow_needs_resize = flow.w < w || flow.h < h;
    const bool can_try_gpu_resize = flow_needs_resize &&
                                    flow_resize_mode != FlowResizeMode::ForceCPU &&
                                    vkdev && rife_flow_resize_output && rife_flow_double_vectors;
    const bool require_gpu_resize = flow_needs_resize && flow_resize_mode == FlowResizeMode::ForceGPU;
    if (can_try_gpu_resize)
    {
        ncnn::VkMat flow_resized_gpu;
        if (rife_flow_resize_output->forward(flow, flow_resized_gpu, cmd, opt) == 0)
        {
            ncnn::VkMat flow_scaled_gpu;
            if (rife_flow_double_vectors->forward(flow_resized_gpu, flow_scaled_gpu, cmd, opt) == 0)
            {
                cmd.record_clone(flow_scaled_gpu, flow_cpu, opt);
                cmd.submit_and_wait();
                used_gpu_resize = true;
            }
        }
    }

    if (!used_gpu_resize)
    {
        if (require_gpu_resize)
        {
            vkdev->reclaim_blob_allocator(blob_vkallocator);
            vkdev->reclaim_staging_allocator(staging_vkallocator);
            return -1;
        }

        cmd.record_clone(flow, flow_cpu, opt);
        cmd.submit_and_wait();
    }

    ncnn::Mat flow_cpu_unpacked;
    if (unpack_flow_channels(flow_cpu, flow_cpu_unpacked) != 0)
    {
        vkdev->reclaim_blob_allocator(blob_vkallocator);
        vkdev->reclaim_staging_allocator(staging_vkallocator);
        return -1;
    }

    int export_status{};
    if (flow_cpu_unpacked.w >= w && flow_cpu_unpacked.h >= h)
        export_status = copy_flow_output_direct(flow_cpu_unpacked, flow_out, w, h);
    else if (flow_cpu_unpacked.w * 2 >= w && flow_cpu_unpacked.h * 2 >= h)
        export_status = copy_flow_output_resized_cpu(flow_cpu_unpacked, flow_out, w, h);
    else
        export_status = -1;
    if (export_status != 0)
    {
        vkdev->reclaim_blob_allocator(blob_vkallocator);
        vkdev->reclaim_staging_allocator(staging_vkallocator);
        return -1;
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

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

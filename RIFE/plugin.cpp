/*
    MIT License

    Copyright (c) 2021-2022 HolyWu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <semaphore>
#include <string>
#include <vector>
#include <iostream>
#include "VapourSynth4.h"
#include "VSHelper4.h"

#include "rife.h"

using namespace std::literals;

static std::atomic<int> numGPUInstances{ 0 };

struct RIFEData;

namespace {

constexpr auto MVToolsAnalysisDataKey = "MVTools_MVAnalysisData";
constexpr auto MVToolsVectorsKey = "MVTools_vectors";
constexpr auto RIFEMVBackwardVectorsInternalKey = "_RIFEMVBackwardVectors";
constexpr auto RIFEMVForwardVectorsInternalKey = "_RIFEMVForwardVectors";
constexpr int MotionIsBackward = 0x00000002;
constexpr int MotionUseChromaMotion = 0x00000008;
constexpr int MVBlockReduceCenter = 0;
constexpr int MVBlockReduceAverage = 1;

using MVArraySizeType = int;

struct MVToolsVector final {
    int x;
    int y;
    int64_t sad;
};

struct MVAnalysisData final {
    int nMagicKey;
    int nVersion;
    int nBlkSizeX;
    int nBlkSizeY;
    int nPel;
    int nLvCount;
    int nDeltaFrame;
    int isBackward;
    int nCPUFlags;
    int nMotionFlags;
    int nWidth;
    int nHeight;
    int nOverlapX;
    int nOverlapY;
    int nBlkX;
    int nBlkY;
    int bitsPerSample;
    int yRatioUV;
    int xRatioUV;
    int nHPadding;
    int nVPadding;
};

struct MotionVectorConfig final {
    bool useChroma;
    int blockSize;
    int overlap;
    int stepX;
    int stepY;
    int pel;
    int bits;
    int hPadding;
    int vPadding;
    int blkX;
    int blkY;
    int blockReduce;
    int64_t invalidSad;
    MVAnalysisData backwardAnalysisData;
    MVAnalysisData forwardAnalysisData;
};

struct ResolvedRIFEModel final {
    std::string modelPath;
    int padding;
    bool rifeV2;
    bool rifeV4;
};

static_assert(sizeof(MVArraySizeType) == 4);
static_assert(sizeof(MVToolsVector) == 16);
static_assert(sizeof(MVAnalysisData) == 84);

static int computeBlockCount(const int size, const int blockSize, const int overlap, const int padding) noexcept {
    const auto step = blockSize - overlap;
    const auto paddedSize = size + padding * 2;

    return std::max(1, (paddedSize - overlap + step - 1) / step);
}

static double rgbToLuma(const float r, const float g, const float b) noexcept {
    return r * 0.2126 + g * 0.7152 + b * 0.0722;
}

static int clampPixel(const int value, const int limit) noexcept {
    return std::clamp(value, 0, limit - 1);
}

static int clampMotionVectorComponent(const int value, const int pel, const int blockCoord,
                                      const int blockSize, const int size, const int padding) noexcept {
    const auto minPixelDelta = -padding - blockCoord;
    const auto maxPixelDelta = size - blockSize + padding - blockCoord;

    return std::clamp(value, minPixelDelta * pel, maxPixelDelta * pel);
}

static MotionVectorConfig createMotionVectorConfig(const VSVideoInfo& inputVi, const VSVideoInfo* const metadataVi,
                                                   const bool useChroma, const int blockSize, const int overlap,
                                                   const int pel, const int bits, const int hPadding,
                                                   const int vPadding, const int blockReduce) noexcept {
    MotionVectorConfig config{};
    config.useChroma = useChroma;
    config.blockSize = blockSize;
    config.overlap = overlap;
    config.stepX = blockSize - overlap;
    config.stepY = blockSize - overlap;
    config.pel = pel;
    config.bits = bits;
    config.hPadding = hPadding;
    config.vPadding = vPadding;
    config.blkX = computeBlockCount(inputVi.width, blockSize, overlap, hPadding);
    config.blkY = computeBlockCount(inputVi.height, blockSize, overlap, vPadding);
    config.blockReduce = blockReduce;
    config.invalidSad = static_cast<int64_t>(blockSize) * blockSize * (1LL << bits);

    const auto& analysisVi = metadataVi ? *metadataVi : inputVi;
    const auto xRatioUV = 1 << analysisVi.format.subSamplingW;
    const auto yRatioUV = 1 << analysisVi.format.subSamplingH;
    const auto makeAnalysisData = [&](const bool backward) {
        MVAnalysisData analysisData{};
        analysisData.nVersion = 5;
        analysisData.nBlkSizeX = config.blockSize;
        analysisData.nBlkSizeY = config.blockSize;
        analysisData.nPel = config.pel;
        analysisData.nLvCount = 1;
        analysisData.nDeltaFrame = 1;
        analysisData.isBackward = backward ? 1 : 0;
        analysisData.nMotionFlags = backward ? MotionIsBackward : 0;
        if (config.useChroma)
            analysisData.nMotionFlags |= MotionUseChromaMotion;
        analysisData.nWidth = inputVi.width;
        analysisData.nHeight = inputVi.height;
        analysisData.nOverlapX = config.overlap;
        analysisData.nOverlapY = config.overlap;
        analysisData.nBlkX = config.blkX;
        analysisData.nBlkY = config.blkY;
        analysisData.bitsPerSample = config.bits;
        analysisData.yRatioUV = yRatioUV;
        analysisData.xRatioUV = xRatioUV;
        analysisData.nHPadding = config.hPadding;
        analysisData.nVPadding = config.vPadding;
        return analysisData;
    };

    config.backwardAnalysisData = makeAnalysisData(true);
    config.forwardAnalysisData = makeAnalysisData(false);
    return config;
}

} // namespace

struct RIFEData final {
    VSNode* node;
    VSNode* psnr;
    VSVideoInfo vi;
    bool exportMotionVectors;
    bool sceneChange;
    bool skip;
    bool mvBackward;
    bool mvUseChroma;
    double skipThreshold;
    int64_t factor;
    int64_t factorNum;
    int64_t factorDen;
    int mvBlockSize;
    int mvOverlap;
    int mvStepX;
    int mvStepY;
    int mvPel;
    int mvBits;
    int mvHPadding;
    int mvVPadding;
    int mvBlkX;
    int mvBlkY;
    int mvBlockReduce;
    int64_t mvInvalidSad;
    MVAnalysisData mvAnalysisData;
    MotionVectorConfig mvConfig;
    std::unique_ptr<RIFE> rife;
    std::unique_ptr<std::counting_semaphore<>> semaphore;
};

static float reduceBlockFlow(const float* flowPlane, const int width, const int height,
                             const int blockX, const int blockY, const RIFEData* const VS_RESTRICT d) noexcept {
    if (d->mvBlockReduce == MVBlockReduceCenter) {
        const auto sampleY = clampPixel(blockY + d->mvBlockSize / 2, height);
        const auto sampleX = clampPixel(blockX + d->mvBlockSize / 2, width);

        return flowPlane[sampleY * width + sampleX];
    }

    double sum{};
    for (auto y = 0; y < d->mvBlockSize; y++) {
        const auto sampleY = clampPixel(blockY + y, height);
        for (auto x = 0; x < d->mvBlockSize; x++) {
            const auto sampleX = clampPixel(blockX + x, width);
            sum += flowPlane[sampleY * width + sampleX];
        }
    }

    return static_cast<float>(sum / static_cast<double>(d->mvBlockSize * d->mvBlockSize));
}

static int64_t computeBlockSAD(const VSFrame* current, const VSFrame* reference, const int pixelDx, const int pixelDy,
                               const int blockX, const int blockY, const int width, const int height,
                               const RIFEData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    const auto stride = vsapi->getStride(current, 0) / vsapi->getVideoFrameFormat(current)->bytesPerSample;
    const auto currentR = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 0));
    const auto currentG = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 1));
    const auto currentB = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 2));
    const auto referenceR = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 0));
    const auto referenceG = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 1));
    const auto referenceB = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 2));
    const auto maxSample = static_cast<double>((1ULL << d->mvBits) - 1ULL);
    int64_t sad{};

    for (auto y = 0; y < d->mvBlockSize; y++) {
        const auto currentY = clampPixel(blockY + y, height);
        const auto referenceY = clampPixel(currentY + pixelDy, height);
        for (auto x = 0; x < d->mvBlockSize; x++) {
            const auto currentX = clampPixel(blockX + x, width);
            const auto referenceX = clampPixel(currentX + pixelDx, width);
            const auto currentIndex = currentY * stride + currentX;
            const auto referenceIndex = referenceY * stride + referenceX;

            if (d->mvUseChroma) {
                sad += static_cast<int64_t>(std::llround((std::abs(currentR[currentIndex] - referenceR[referenceIndex]) +
                                                          std::abs(currentG[currentIndex] - referenceG[referenceIndex]) +
                                                          std::abs(currentB[currentIndex] - referenceB[referenceIndex])) * maxSample));
            } else {
                sad += static_cast<int64_t>(std::llround(std::abs(rgbToLuma(currentR[currentIndex], currentG[currentIndex], currentB[currentIndex]) -
                                                                 rgbToLuma(referenceR[referenceIndex], referenceG[referenceIndex], referenceB[referenceIndex])) * maxSample));
            }
        }
    }

    return sad;
}

static std::vector<char> buildMVToolsVectorBlob(const VSFrame* current, const VSFrame* reference, const float* flow,
                                                const bool valid, const RIFEData* const VS_RESTRICT d,
                                                const VSAPI* vsapi) {
    const auto vectorCount = static_cast<size_t>(d->mvBlkX) * d->mvBlkY;
    std::vector<MVToolsVector> vectors(vectorCount);

    if (!valid) {
        for (auto& vector : vectors) {
            vector.x = 0;
            vector.y = 0;
            vector.sad = d->mvInvalidSad;
        }
    } else {
        const auto width = vsapi->getFrameWidth(current, 0);
        const auto height = vsapi->getFrameHeight(current, 0);
        const auto channelOffset = d->mvBackward ? 0 : 2;
        const auto flowPlaneSize = width * height;

        for (auto by = 0; by < d->mvBlkY; by++) {
            const auto blockY = by * d->mvStepY - d->mvVPadding;
            for (auto bx = 0; bx < d->mvBlkX; bx++) {
                const auto blockX = bx * d->mvStepX - d->mvHPadding;
                auto& vector = vectors[static_cast<size_t>(by) * d->mvBlkX + bx];
                const auto flowX = reduceBlockFlow(flow + (channelOffset + 0) * flowPlaneSize, width, height, blockX, blockY, d);
                const auto flowY = reduceBlockFlow(flow + (channelOffset + 1) * flowPlaneSize, width, height, blockX, blockY, d);

                vector.x = static_cast<int>(std::lround(-2.0f * flowX * d->mvPel));
                vector.y = static_cast<int>(std::lround(-2.0f * flowY * d->mvPel));
                vector.x = clampMotionVectorComponent(vector.x, d->mvPel, blockX, d->mvBlockSize, width, d->mvHPadding);
                vector.y = clampMotionVectorComponent(vector.y, d->mvPel, blockY, d->mvBlockSize, height, d->mvVPadding);
                vector.sad = computeBlockSAD(current, reference,
                                             static_cast<int>(std::lround(static_cast<double>(vector.x) / d->mvPel)),
                                             static_cast<int>(std::lround(static_cast<double>(vector.y) / d->mvPel)),
                                             blockX, blockY, width, height, d, vsapi);
            }
        }
    }

    const auto planeSize = static_cast<MVArraySizeType>(sizeof(MVArraySizeType) + vectors.size() * sizeof(MVToolsVector));
    const auto groupSize = static_cast<MVArraySizeType>(sizeof(MVArraySizeType) * 2 + planeSize);
    std::vector<char> blob(groupSize);
    size_t offset{};
    const auto writeScalar = [&](const auto value) {
        std::memcpy(blob.data() + offset, &value, sizeof(value));
        offset += sizeof(value);
    };

    writeScalar(groupSize);
    writeScalar(valid ? MVArraySizeType{ 1 } : MVArraySizeType{ 0 });
    writeScalar(planeSize);
    std::memcpy(blob.data() + offset, vectors.data(), vectors.size() * sizeof(MVToolsVector));

    return blob;
}

static void filter(const VSFrame* src0, const VSFrame* src1, VSFrame* dst,
                   const float timestep, const RIFEData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    const auto width{ vsapi->getFrameWidth(src0, 0) };
    const auto height{ vsapi->getFrameHeight(src0, 0) };
    const auto stride{ vsapi->getStride(src0, 0) / d->vi.format.bytesPerSample };
    auto src0R{ reinterpret_cast<const float*>(vsapi->getReadPtr(src0, 0)) };
    auto src0G{ reinterpret_cast<const float*>(vsapi->getReadPtr(src0, 1)) };
    auto src0B{ reinterpret_cast<const float*>(vsapi->getReadPtr(src0, 2)) };
    auto src1R{ reinterpret_cast<const float*>(vsapi->getReadPtr(src1, 0)) };
    auto src1G{ reinterpret_cast<const float*>(vsapi->getReadPtr(src1, 1)) };
    auto src1B{ reinterpret_cast<const float*>(vsapi->getReadPtr(src1, 2)) };
    auto dstR{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0)) };
    auto dstG{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, 1)) };
    auto dstB{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, 2)) };

    d->semaphore->acquire();
    d->rife->process(src0R, src0G, src0B, src1R, src1G, src1B, dstR, dstG, dstB, width, height, stride, timestep);
    d->semaphore->release();
}

static bool attachMotionVectors(const VSFrame* current, const VSFrame* reference, VSFrame* dst,
                                const RIFEData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    const auto width = vsapi->getFrameWidth(current, 0);
    const auto height = vsapi->getFrameHeight(current, 0);
    const auto stride = vsapi->getStride(current, 0) / vsapi->getVideoFrameFormat(current)->bytesPerSample;
    auto props = vsapi->getFramePropertiesRW(dst);
    std::vector<char> vectorBlob;

    if (reference) {
        std::vector<float> flow(static_cast<size_t>(width) * height * 4);
        const auto first = d->mvBackward ? current : reference;
        const auto second = d->mvBackward ? reference : current;
        const auto firstR = reinterpret_cast<const float*>(vsapi->getReadPtr(first, 0));
        const auto firstG = reinterpret_cast<const float*>(vsapi->getReadPtr(first, 1));
        const auto firstB = reinterpret_cast<const float*>(vsapi->getReadPtr(first, 2));
        const auto secondR = reinterpret_cast<const float*>(vsapi->getReadPtr(second, 0));
        const auto secondG = reinterpret_cast<const float*>(vsapi->getReadPtr(second, 1));
        const auto secondB = reinterpret_cast<const float*>(vsapi->getReadPtr(second, 2));

        d->semaphore->acquire();
        const auto status = d->rife->process_flow(firstR, firstG, firstB, secondR, secondG, secondB, flow.data(), width, height, stride);
        d->semaphore->release();
        if (status != 0)
            return false;

        vectorBlob = buildMVToolsVectorBlob(current, reference, flow.data(), true, d, vsapi);
    } else {
        vectorBlob = buildMVToolsVectorBlob(current, current, nullptr, false, d, vsapi);
    }

    vsapi->mapSetData(props, MVToolsAnalysisDataKey, reinterpret_cast<const char*>(&d->mvAnalysisData), sizeof(d->mvAnalysisData), dtBinary, maReplace);
    vsapi->mapSetData(props, MVToolsVectorsKey, vectorBlob.data(), static_cast<int>(vectorBlob.size()), dtBinary, maReplace);

    return true;
}

static const VSFrame* VS_CC rifeGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData,
                                         VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEData*>(instanceData) };

    if (d->exportMotionVectors) {
        if (activationReason == arInitial) {
            vsapi->requestFrameFilter(n, d->node, frameCtx);
            if (d->mvBackward) {
                if (n + 1 < d->vi.numFrames)
                    vsapi->requestFrameFilter(n + 1, d->node, frameCtx);
            } else if (n > 0) {
                vsapi->requestFrameFilter(n - 1, d->node, frameCtx);
            }
        } else if (activationReason == arAllFramesReady) {
            auto current = vsapi->getFrameFilter(n, d->node, frameCtx);
            const VSFrame* reference{};
            if (d->mvBackward) {
                if (n + 1 < d->vi.numFrames)
                    reference = vsapi->getFrameFilter(n + 1, d->node, frameCtx);
            } else if (n > 0) {
                reference = vsapi->getFrameFilter(n - 1, d->node, frameCtx);
            }

            auto dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, current, core);
            auto* dstp = vsapi->getWritePtr(dst, 0);
            const auto dstStride = vsapi->getStride(dst, 0);
            for (auto y = 0; y < d->vi.height; y++)
                std::memset(dstp + static_cast<size_t>(y) * dstStride, 0, d->vi.width * d->vi.format.bytesPerSample);

            if (!attachMotionVectors(current, reference, dst, d, vsapi)) {
                vsapi->freeFrame(current);
                vsapi->freeFrame(reference);
                vsapi->freeFrame(dst);
                vsapi->setFilterError("RIFE: failed to export motion vectors", frameCtx);
                return nullptr;
            }

            vsapi->freeFrame(current);
            vsapi->freeFrame(reference);
            return dst;
        }

        return nullptr;
    }

    auto frameNum{ static_cast<int>(n * d->factorDen / d->factorNum) };
    auto remainder{ n * d->factorDen % d->factorNum };

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(frameNum, d->node, frameCtx);
        if (remainder != 0 && n < d->vi.numFrames - d->factor)
            vsapi->requestFrameFilter(frameNum + 1, d->node, frameCtx);

        if (d->skip)
            vsapi->requestFrameFilter(frameNum, d->psnr, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto src0{ vsapi->getFrameFilter(frameNum, d->node, frameCtx) };
        decltype(src0) src1{};
        decltype(src0) psnr{};
        VSFrame* dst{};

        if (remainder != 0 && n < d->vi.numFrames - d->factor) {
            bool sceneChange{};
            double psnrY{ -1.0 };
            int err;

            if (d->sceneChange)
                sceneChange = !!vsapi->mapGetInt(vsapi->getFramePropertiesRO(src0), "_SceneChangeNext", 0, &err);

            if (d->skip) {
                psnr = vsapi->getFrameFilter(frameNum, d->psnr, frameCtx);
                psnrY = vsapi->mapGetFloat(vsapi->getFramePropertiesRO(psnr), "psnr_y", 0, nullptr);
            }

            if (sceneChange || psnrY >= d->skipThreshold) {
                dst = vsapi->copyFrame(src0, core);
            } else {
                src1 = vsapi->getFrameFilter(frameNum + 1, d->node, frameCtx);
                dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src0, core);
                filter(src0, src1, dst, static_cast<float>(remainder) / d->factorNum, d, vsapi);
            }
        } else {
            dst = vsapi->copyFrame(src0, core);
        }

        auto props{ vsapi->getFramePropertiesRW(dst) };
        int errNum, errDen;
        auto durationNum{ vsapi->mapGetInt(props, "_DurationNum", 0, &errNum) };
        auto durationDen{ vsapi->mapGetInt(props, "_DurationDen", 0, &errDen) };
        if (!errNum && !errDen) {
            vsh::muldivRational(&durationNum, &durationDen, d->factorDen, d->factorNum);
            vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
            vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
        }

        vsapi->freeFrame(src0);
        vsapi->freeFrame(src1);
        vsapi->freeFrame(psnr);
        return dst;
    }

    return nullptr;
}

static void VS_CC rifeFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<RIFEData*>(instanceData) };
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->psnr);
    delete d;

    if (--numGPUInstances == 0)
        ncnn::destroy_gpu_instance();
}

static void VS_CC rifeCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d{ std::make_unique<RIFEData>() };
    VSNode* mvClip{};

    try {
        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = *vsapi->getVideoInfo(d->node);
        VSVideoInfo mvClipVi{};
        bool hasMVClip{};
        int err;

        if (!vsh::isConstantVideoFormat(&d->vi) ||
            d->vi.format.colorFamily != cfRGB ||
            d->vi.format.sampleType != stFloat ||
            d->vi.format.bitsPerSample != 32)
            throw "only constant RGB format 32 bit float input supported";

        if (ncnn::create_gpu_instance())
            throw "failed to create GPU instance";
        ++numGPUInstances;

        auto model{ vsapi->mapGetIntSaturated(in, "model", 0, &err) };
        if (err)
            model = 5;

        auto factorNum{ vsapi->mapGetInt(in, "factor_num", 0, &err) };
        if (err)
            factorNum = 2;

        auto factorDen{ vsapi->mapGetInt(in, "factor_den", 0, &err) };
        if (err)
            factorDen = 1;

        auto fpsNum{ vsapi->mapGetInt(in, "fps_num", 0, &err) };
        if (!err && fpsNum < 1)
            throw "fps_num must be at least 1";

        auto fpsDen{ vsapi->mapGetInt(in, "fps_den", 0, &err) };
        if (!err && fpsDen < 1)
            throw "fps_den must be at least 1";

        auto model_path{ vsapi->mapGetData(in, "model_path", 0, &err) };
        std::string modelPath{ err ? "" : model_path };

        auto gpuId{ vsapi->mapGetIntSaturated(in, "gpu_id", 0, &err) };
        if (err)
            gpuId = ncnn::get_default_gpu_index();

        auto gpuThread{ vsapi->mapGetIntSaturated(in, "gpu_thread", 0, &err) };
        if (err)
            gpuThread = 2;

        auto tta{ !!vsapi->mapGetInt(in, "tta", 0, &err) };
        auto uhd{ !!vsapi->mapGetInt(in, "uhd", 0, &err) };
        d->exportMotionVectors = !!vsapi->mapGetInt(in, "mv", 0, &err);
        d->mvBackward = !!vsapi->mapGetInt(in, "mv_backward", 0, &err);
        if (err)
            d->mvBackward = true;
        auto mvBlockSize{ vsapi->mapGetIntSaturated(in, "mv_block_size", 0, &err) };
        if (err)
            mvBlockSize = 16;
        auto mvOverlap{ vsapi->mapGetIntSaturated(in, "mv_overlap", 0, &err) };
        if (err)
            mvOverlap = 8;
        auto mvPel{ vsapi->mapGetIntSaturated(in, "mv_pel", 0, &err) };
        if (err)
            mvPel = 1;
        auto mvBits{ vsapi->mapGetIntSaturated(in, "mv_bits", 0, &err) };
        const auto mvBitsSpecified = !err;
        if (err)
            mvBits = 8;
        auto mvHPadding{ vsapi->mapGetIntSaturated(in, "mv_hpad", 0, &err) };
        if (err)
            mvHPadding = 0;
        auto mvVPadding{ vsapi->mapGetIntSaturated(in, "mv_vpad", 0, &err) };
        if (err)
            mvVPadding = 0;
        auto mvBlockReduce{ vsapi->mapGetIntSaturated(in, "mv_block_reduce", 0, &err) };
        if (err)
            mvBlockReduce = MVBlockReduceAverage;
        d->mvUseChroma = !!vsapi->mapGetInt(in, "mv_chroma", 0, &err);
        mvClip = vsapi->mapGetNode(in, "mv_clip", 0, &err);
        if (!err) {
            mvClipVi = *vsapi->getVideoInfo(mvClip);
            hasMVClip = true;
        }
        d->sceneChange = !!vsapi->mapGetInt(in, "sc", 0, &err);
        d->skip = !!vsapi->mapGetInt(in, "skip", 0, &err);

        d->skipThreshold = vsapi->mapGetFloat(in, "skip_threshold", 0, &err);
        if (err)
            d->skipThreshold = 60.0;

        if (model < 0 || model > 76)
            throw "model must be between 0 and 76 (inclusive)";

        if (factorNum < 1)
            throw "factor_num must be at least 1";

        if (factorDen < 1)
            throw "factor_den must be at least 1";

        if (hasMVClip) {
            if (!vsh::isConstantVideoFormat(&mvClipVi))
                throw "mv_clip must have a constant format";

            if (mvClipVi.width != d->vi.width || mvClipVi.height != d->vi.height)
                throw "mv_clip dimensions must match clip";

            if (!mvBitsSpecified)
                mvBits = mvClipVi.format.bitsPerSample;
        }

        if (fpsNum && fpsDen && !(d->vi.fpsNum && d->vi.fpsDen))
            throw "clip does not have a valid frame rate and hence fps_num and fps_den cannot be used";

        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count())
            throw "invalid GPU device";

        if (auto queueCount{ ncnn::get_gpu_info(gpuId).compute_queue_count() }; static_cast<uint32_t>(gpuThread) > queueCount)
            std::cerr << "Warning: gpu_thread is recommended to be between 1 and " << queueCount << " (inclusive)" << std::endl;
        
        if (auto queueCount{ ncnn::get_gpu_info(gpuId).compute_queue_count() }; gpuThread < 1)
            throw "gpu_thread must be greater than 0";

        
        if (d->skipThreshold < 0 || d->skipThreshold > 60)
            throw "skip_threshold must be between 0.0 and 60.0 (inclusive)";

        if (d->exportMotionVectors) {
            if (fpsNum || fpsDen || factorNum != 2 || factorDen != 1)
                throw "mv=True does not support factor_num, factor_den, fps_num, or fps_den";

            d->factorNum = 1;
            d->factorDen = 1;
        } else if (fpsNum && fpsDen) {
            vsh::muldivRational(&fpsNum, &fpsDen, d->vi.fpsDen, d->vi.fpsNum);
            d->factorNum = fpsNum;
            d->factorDen = fpsDen;
        } else {
            d->factorNum = factorNum;
            d->factorDen = factorDen;
        }
        vsh::muldivRational(&d->vi.fpsNum, &d->vi.fpsDen, d->factorNum, d->factorDen);

        if (d->vi.numFrames < 2)
            throw "clip's number of frames must be at least 2";

        if (d->vi.numFrames / d->factorDen > INT_MAX / d->factorNum)
            throw "resulting clip is too long";

        auto oldNumFrames{ d->vi.numFrames };
        d->vi.numFrames = static_cast<int>(d->vi.numFrames * d->factorNum / d->factorDen);

        d->factor = d->factorNum / d->factorDen;

        if (!!vsapi->mapGetInt(in, "list_gpu", 0, &err)) {
            std::string text;

            for (auto i{ 0 }; i < ncnn::get_gpu_count(); i++)
                text += std::to_string(i) + ": " + ncnn::get_gpu_info(i).device_name() + "\n";

            auto args{ vsapi->createMap() };
            vsapi->mapConsumeNode(args, "clip", d->node, maReplace);
            vsapi->mapSetData(args, "text", text.c_str(), -1, dtUtf8, maReplace);

            auto ret{ vsapi->invoke(vsapi->getPluginByID(VSH_TEXT_PLUGIN_ID, core), "Text", args) };
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);

                if (--numGPUInstances == 0)
                    ncnn::destroy_gpu_instance();
                return;
            }

            vsapi->mapConsumeNode(out, "clip", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
            vsapi->freeMap(args);
            vsapi->freeMap(ret);

            if (--numGPUInstances == 0)
                ncnn::destroy_gpu_instance();
            return;
        }
        int padding;
        padding = 32;
        if (modelPath.empty()) {
            std::string pluginPath{ vsapi->getPluginPath(vsapi->getPluginByID("com.holywu.rife", core)) };
            modelPath = pluginPath.substr(0, pluginPath.rfind('/')) + "/models";

            switch (model) {
            case 0:
                modelPath += "/rife";
                break;
            case 1:
                modelPath += "/rife-HD";
                break;
            case 2:
                modelPath += "/rife-UHD";
                break;
            case 3:
                modelPath += "/rife-anime";
                break;
            case 4:
                modelPath += "/rife-v2";
                break;
            case 5:
                modelPath += "/rife-v2.3";
                break;
            case 6:
                modelPath += "/rife-v2.4";
                break;
            case 7:
                modelPath += "/rife-v3.0";
                break;
            case 8:
                modelPath += "/rife-v3.1";
                break;
            
            case 9:
                modelPath += "/rife-v3.9_ensembleFalse_fastTrue";
                break;
            case 10:
                modelPath += "/rife-v3.9_ensembleTrue_fastFalse";
                break;
            case 11:
                modelPath += "/rife-v4_ensembleFalse_fastTrue";
                break;
            case 12:
                modelPath += "/rife-v4_ensembleTrue_fastFalse";
                break;
            case 13:
                modelPath += "/rife-v4.1_ensembleFalse_fastTrue";
                break;
            case 14:
                modelPath += "/rife-v4.1_ensembleTrue_fastFalse";
                break;
            case 15:
                modelPath += "/rife-v4.2_ensembleFalse_fastTrue";
                break;
            case 16:
                modelPath += "/rife-v4.2_ensembleTrue_fastFalse";
                break;
            case 17:
                modelPath += "/rife-v4.3_ensembleFalse_fastTrue";
                break;
            case 18:
                modelPath += "/rife-v4.3_ensembleTrue_fastFalse";
                break;
            case 19:
                modelPath += "/rife-v4.4_ensembleFalse_fastTrue";
                break;
            case 20:
                modelPath += "/rife-v4.4_ensembleTrue_fastFalse";
                break;
            case 21:
                modelPath += "/rife-v4.5_ensembleFalse";
                break;
            case 22:
                modelPath += "/rife-v4.5_ensembleTrue";
                break;
            case 23:
                modelPath += "/rife-v4.6_ensembleFalse";
                break;
            case 24:
                modelPath += "/rife-v4.6_ensembleTrue";
                break;
            case 25:
                modelPath += "/rife-v4.7_ensembleFalse";
                break;
            case 26:
                modelPath += "/rife-v4.7_ensembleTrue";
                break;
            case 27:
                modelPath += "/rife-v4.8_ensembleFalse";
                break;
            case 28:
                modelPath += "/rife-v4.8_ensembleTrue";
                break;
            case 29:
                modelPath += "/rife-v4.9_ensembleFalse";
                break;
            case 30:
                modelPath += "/rife-v4.9_ensembleTrue";
                break;
            case 31:
                modelPath += "/rife-v4.10_ensembleFalse";
                break;
            case 32:
                modelPath += "/rife-v4.10_ensembleTrue";
                break;
            case 33:
                modelPath += "/rife-v4.11_ensembleFalse";
                break;
            case 34:
                modelPath += "/rife-v4.11_ensembleTrue";
                break;
            case 35:
                modelPath += "/rife-v4.12_ensembleFalse";
                break;
            case 36:
                modelPath += "/rife-v4.12_ensembleTrue";
                break;
            case 37:
                modelPath += "/rife-v4.12_lite_ensembleFalse";
                break;
            case 38:
                modelPath += "/rife-v4.12_lite_ensembleTrue";
                break;
            case 39:
                modelPath += "/rife-v4.13_ensembleFalse";
                break;
            case 40:
                modelPath += "/rife-v4.13_ensembleTrue";
                break;
            case 41:
                modelPath += "/rife-v4.13_lite_ensembleFalse";
                break;
            case 42:
                modelPath += "/rife-v4.13_lite_ensembleTrue";
                break;
            case 43:
                modelPath += "/rife-v4.14_ensembleFalse";
                break;
            case 44:
                modelPath += "/rife-v4.14_ensembleTrue";
                break;
            case 45:
                modelPath += "/rife-v4.14_lite_ensembleFalse";
                break;
            case 46:
                modelPath += "/rife-v4.14_lite_ensembleTrue";
                break;
            case 47:
                modelPath += "/rife-v4.15_ensembleFalse";
                break;
            case 48:
                modelPath += "/rife-v4.15_ensembleTrue";
                break;
            case 49:
                modelPath += "/rife-v4.15_lite_ensembleFalse";
                break;
            case 50:
                modelPath += "/rife-v4.15_lite_ensembleTrue";
                break;
            case 51:
                modelPath += "/rife-v4.16_lite_ensembleFalse";
                break;
            case 52:
                modelPath += "/rife-v4.16_lite_ensembleTrue";
                break;
            case 53:
                modelPath += "/rife-v4.17_ensembleFalse";
                break;
            case 54:
                modelPath += "/rife-v4.17_ensembleTrue";
                break;
            case 55:
                modelPath += "/rife-v4.17_lite_ensembleFalse";
                break;
            case 56:
                modelPath += "/rife-v4.17_lite_ensembleTrue";
                break;
            case 57:
                modelPath += "/rife-v4.18_ensembleFalse";
                break;
            case 58:
                modelPath += "/rife-v4.18_ensembleTrue";
                break;
            case 59:
                modelPath += "/rife-v4.19_beta_ensembleFalse";
                break;
            case 60:
                modelPath += "/rife-v4.19_beta_ensembleTrue";
                break;
            case 61:
                modelPath += "/rife-v4.20_ensembleFalse";
                break;
            case 62:
                modelPath += "/rife-v4.20_ensembleTrue";
                break;
            case 63:
                modelPath += "/rife-v4.21_ensembleFalse";
                break;
            case 64:
                modelPath += "/rife-v4.22_ensembleFalse";
                break;
            case 65:
                modelPath += "/rife-v4.22_lite_ensembleFalse";
                break;
            case 66:
                modelPath += "/rife-v4.23_beta_ensembleFalse";
                break;
            case 67:
                modelPath += "/rife-v4.24_ensembleFalse";
                break;
            case 68:
                modelPath += "/rife-v4.24_ensembleTrue";
                break;
            case 69:
                modelPath += "/rife-v4.25_ensembleFalse";
                padding = 64;
                break;
            case 70:
                modelPath += "/rife-v4.25-lite_ensembleFalse";
                padding = 128;
                break;
            case 71:
                modelPath += "/rife-v4.25_heavy_beta_ensembleFalse";
                padding = 64;
                break;
            case 72:
                modelPath += "/rife-v4.26_ensembleFalse";
                padding = 64;
                break;
            case 73:
                modelPath += "/rife-v4.26-large_ensembleFalse";
                padding = 64;
                break;
            }
        }

        std::ifstream ifs{ modelPath + "/flownet.param" };
        if (!ifs.is_open())
            throw "failed to load model";
        ifs.close();

        bool rife_v2{};
        bool rife_v4{};
        
        if (modelPath.find("rife-v2") != std::string::npos)
            rife_v2 = true;
        else if (modelPath.find("rife-v3.9") != std::string::npos)
            rife_v4 = true;
        
        else if (modelPath.find("rife-v3") != std::string::npos)
            rife_v2 = true;
        else if (modelPath.find("rife-v4") != std::string::npos)
            rife_v4 = true;
        else if (modelPath.find("rife4") != std::string::npos)
            rife_v4 = true;
        // rife 4.25 and 4.26 require more padding due to extra scales.
        if (modelPath.find("rife-v4.25") != std::string::npos)
            padding = 64;
        if (modelPath.find("rife-v4.25-lite") != std::string::npos) 
            padding = 128;
        if (modelPath.find("rife-v4.26") != std::string::npos)
            padding = 64;
        else if (modelPath.find("rife") == std::string::npos)
            throw "unknown model dir type";

        if (!d->exportMotionVectors && !rife_v4 && (d->factorNum != 2 || d->factorDen != 1))
            throw "only rife-v4 model supports custom frame rate";

        if (rife_v4 && tta)
            throw "rife-v4 model does not support TTA mode";

        if (d->exportMotionVectors) {
            if (tta)
                throw "mv=True does not support TTA mode";

            if (d->sceneChange || d->skip)
                throw "mv=True does not support sc or skip";

            if (modelPath.find("rife-v3.1") == std::string::npos)
                throw "mv=True currently requires the rife-v3.1 model";

            if (mvBlockSize < 1)
                throw "mv_block_size must be at least 1";

            if (mvOverlap < 0 || mvOverlap >= mvBlockSize)
                throw "mv_overlap must be between 0 and mv_block_size - 1";

            if (mvPel < 1)
                throw "mv_pel must be at least 1";

            if (mvBits < 1 || mvBits > 16)
                throw "mv_bits must be between 1 and 16 (inclusive)";

            if (mvHPadding < 0 || mvVPadding < 0)
                throw "mv_hpad and mv_vpad must be non-negative";

            if (mvBlockReduce != MVBlockReduceCenter && mvBlockReduce != MVBlockReduceAverage)
                throw "mv_block_reduce must be 0 (center) or 1 (average)";

            d->mvBlockSize = mvBlockSize;
            d->mvOverlap = mvOverlap;
            d->mvStepX = mvBlockSize - mvOverlap;
            d->mvStepY = mvBlockSize - mvOverlap;
            d->mvPel = mvPel;
            d->mvBits = mvBits;
            d->mvHPadding = mvHPadding;
            d->mvVPadding = mvVPadding;
            d->mvBlkX = computeBlockCount(d->vi.width, d->mvBlockSize, d->mvOverlap, d->mvHPadding);
            d->mvBlkY = computeBlockCount(d->vi.height, d->mvBlockSize, d->mvOverlap, d->mvVPadding);
            d->mvBlockReduce = mvBlockReduce;
            d->mvInvalidSad = static_cast<int64_t>(d->mvBlockSize) * d->mvBlockSize * (1LL << d->mvBits);
            const auto xRatioUV = hasMVClip ? 1 << mvClipVi.format.subSamplingW : 1 << d->vi.format.subSamplingW;
            const auto yRatioUV = hasMVClip ? 1 << mvClipVi.format.subSamplingH : 1 << d->vi.format.subSamplingH;
            d->mvAnalysisData = {};
            d->mvAnalysisData.nVersion = 5;
            d->mvAnalysisData.nBlkSizeX = d->mvBlockSize;
            d->mvAnalysisData.nBlkSizeY = d->mvBlockSize;
            d->mvAnalysisData.nPel = d->mvPel;
            d->mvAnalysisData.nLvCount = 1;
            d->mvAnalysisData.nDeltaFrame = 1;
            d->mvAnalysisData.isBackward = d->mvBackward ? 1 : 0;
            d->mvAnalysisData.nMotionFlags = d->mvBackward ? MotionIsBackward : 0;
            if (d->mvUseChroma)
                d->mvAnalysisData.nMotionFlags |= MotionUseChromaMotion;
            d->mvAnalysisData.nWidth = d->vi.width;
            d->mvAnalysisData.nHeight = d->vi.height;
            d->mvAnalysisData.nOverlapX = d->mvOverlap;
            d->mvAnalysisData.nOverlapY = d->mvOverlap;
            d->mvAnalysisData.nBlkX = d->mvBlkX;
            d->mvAnalysisData.nBlkY = d->mvBlkY;
            d->mvAnalysisData.bitsPerSample = d->mvBits;
            d->mvAnalysisData.yRatioUV = yRatioUV;
            d->mvAnalysisData.xRatioUV = xRatioUV;
            d->mvAnalysisData.nHPadding = d->mvHPadding;
            d->mvAnalysisData.nVPadding = d->mvVPadding;

            if (!vsapi->getVideoFormatByID(&d->vi.format, pfGray8, core))
                throw "failed to create mv=True output format";
        }

        if (mvClip) {
            vsapi->freeNode(mvClip);
            mvClip = nullptr;
        }

        d->semaphore = std::make_unique<std::counting_semaphore<>>(gpuThread);

        if (d->skip) {
            auto vmaf{ vsapi->getPluginByID("com.holywu.vmaf", core) };

            if (!vmaf)
                throw "VMAF plugin is required when skip=True";

            auto args{ vsapi->createMap() };
            vsapi->mapConsumeNode(args, "clip", d->node, maReplace);
            vsapi->mapSetInt(args, "width", std::min(d->vi.width, 512), maReplace);
            vsapi->mapSetInt(args, "height", std::min(d->vi.height, 512), maReplace);
            vsapi->mapSetInt(args, "format", pfYUV420P8, maReplace);
            vsapi->mapSetData(args, "matrix_s", "709", -1, dtUtf8, maReplace);

            auto ret{ vsapi->invoke(vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core), "Bicubic", args) };
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);

                if (--numGPUInstances == 0)
                    ncnn::destroy_gpu_instance();
                return;
            }

            vsapi->clearMap(args);
            auto reference{ vsapi->mapGetNode(ret, "clip", 0, nullptr) };
            vsapi->mapSetNode(args, "clip", reference, maReplace);
            vsapi->mapSetInt(args, "frames", oldNumFrames - 1, maReplace);

            vsapi->freeMap(ret);
            ret = vsapi->invoke(vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "DuplicateFrames", args);
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);

                if (--numGPUInstances == 0)
                    ncnn::destroy_gpu_instance();
                return;
            }

            vsapi->clearMap(args);
            vsapi->mapConsumeNode(args, "clip", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
            vsapi->mapSetInt(args, "first", 1, maReplace);

            vsapi->freeMap(ret);
            ret = vsapi->invoke(vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "Trim", args);
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);

                if (--numGPUInstances == 0)
                    ncnn::destroy_gpu_instance();
                return;
            }

            vsapi->clearMap(args);
            vsapi->mapConsumeNode(args, "reference", reference, maReplace);
            vsapi->mapConsumeNode(args, "distorted", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
            vsapi->mapSetInt(args, "feature", 0, maReplace);

            vsapi->freeMap(ret);
            ret = vsapi->invoke(vmaf, "Metric", args);
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);

                if (--numGPUInstances == 0)
                    ncnn::destroy_gpu_instance();
                return;
            }

            d->psnr = vsapi->mapGetNode(ret, "clip", 0, nullptr);
            vsapi->freeMap(args);
            vsapi->freeMap(ret);
        }

        d->rife = std::make_unique<RIFE>(gpuId, tta, uhd, 1, rife_v2, rife_v4, padding);

#ifdef _WIN32
        auto bufferSize{ MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, nullptr, 0) };
        std::vector<wchar_t> wbuffer(bufferSize);
        MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, wbuffer.data(), bufferSize);
        d->rife->load(wbuffer.data());
#else
        d->rife->load(modelPath);
#endif
    } catch (const char* error) {
        vsapi->mapSetError(out, ("RIFE: "s + error).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->psnr);
        vsapi->freeNode(mvClip);

        if (--numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }

    std::vector<VSFilterDependency> deps{ {d->node, rpGeneral} };
    if (d->skip)
        deps.push_back({ d->psnr, rpGeneral });
    vsapi->createVideoFilter(out, "RIFE", &d->vi, rifeGetFrame, rifeFree, fmParallel, deps.data(), static_cast<int>(deps.size()), d.get(), core);
    d.release();
}

static void VS_CC rifeMVCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto* plugin = vsapi->getPluginByID("com.holywu.rife", core);
    if (!plugin) {
        vsapi->mapSetError(out, "RIFEMV: failed to find RIFE plugin");
        return;
    }

    auto* args = vsapi->createMap();
    auto* ret = static_cast<VSMap*>(nullptr);
    VSNode* bwNode{};
    VSNode* fwNode{};

    vsapi->copyMap(in, args);
    vsapi->mapSetInt(args, "mv", 1, maReplace);

    vsapi->mapSetInt(args, "mv_backward", 1, maReplace);
    ret = vsapi->invoke(plugin, "RIFE", args);
    if (vsapi->mapGetError(ret)) {
        vsapi->mapSetError(out, vsapi->mapGetError(ret));
        vsapi->freeMap(ret);
        vsapi->freeMap(args);
        return;
    }
    bwNode = vsapi->mapGetNode(ret, "clip", 0, nullptr);
    vsapi->freeMap(ret);

    vsapi->mapSetInt(args, "mv_backward", 0, maReplace);
    ret = vsapi->invoke(plugin, "RIFE", args);
    if (vsapi->mapGetError(ret)) {
        vsapi->mapSetError(out, vsapi->mapGetError(ret));
        vsapi->freeNode(bwNode);
        vsapi->freeMap(ret);
        vsapi->freeMap(args);
        return;
    }
    fwNode = vsapi->mapGetNode(ret, "clip", 0, nullptr);
    vsapi->freeMap(ret);
    vsapi->freeMap(args);

    vsapi->mapConsumeNode(out, "clip", bwNode, maAppend);
    vsapi->mapConsumeNode(out, "clip", fwNode, maAppend);
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.rife", "rife", "Real-Time Intermediate Flow Estimation for Video Frame Interpolation",
                         VS_MAKE_VERSION(9, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    vspapi->registerFunction("RIFE",
                             "clip:vnode;"
                             "model:int:opt;"
                             "factor_num:int:opt;"
                             "factor_den:int:opt;"
                             "fps_num:int:opt;"
                             "fps_den:int:opt;"
                             "model_path:data:opt;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "tta:int:opt;"
                             "uhd:int:opt;"
                             "mv:int:opt;"
                             "mv_backward:int:opt;"
                             "mv_block_size:int:opt;"
                             "mv_overlap:int:opt;"
                             "mv_pel:int:opt;"
                             "mv_bits:int:opt;"
                             "mv_clip:vnode:opt;"
                             "mv_hpad:int:opt;"
                             "mv_vpad:int:opt;"
                             "mv_block_reduce:int:opt;"
                             "mv_chroma:int:opt;"
                             "sc:int:opt;"
                             "skip:int:opt;"
                             "skip_threshold:float:opt;"
                             "list_gpu:int:opt;",
                             "clip:vnode;",
                             rifeCreate, nullptr, plugin);

    vspapi->registerFunction("RIFEMV",
                             "clip:vnode;"
                             "model:int:opt;"
                             "model_path:data:opt;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "uhd:int:opt;"
                             "mv_block_size:int:opt;"
                             "mv_overlap:int:opt;"
                             "mv_pel:int:opt;"
                             "mv_bits:int:opt;"
                             "mv_clip:vnode:opt;"
                             "mv_hpad:int:opt;"
                             "mv_vpad:int:opt;"
                             "mv_block_reduce:int:opt;"
                             "mv_chroma:int:opt;",
                             "clip:vnode[];",
                             rifeMVCreate, nullptr, plugin);
}

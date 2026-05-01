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
constexpr auto RIFEMVBackwardDisplacementInternalKey = "_RIFEMVBackwardDisplacement";
constexpr auto RIFEMVForwardDisplacementInternalKey = "_RIFEMVForwardDisplacement";
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
    int delta;
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

constexpr auto RIFEMVModelRequirementError =
    "motion-vector export requires the rife-v3.1 model or a rife-v4.2+ model";
constexpr auto RIFEMVUnsupportedEarlyV4Error =
    "legacy rife-v4, rife-v4.0, and rife-v4.1 are not supported for motion-vector export; use rife-v4.2 or newer";

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
                                                   const int pel, const int delta, const int bits, const int hPadding,
                                                   const int vPadding, const int blockReduce) noexcept {
    MotionVectorConfig config{};
    config.useChroma = useChroma;
    config.blockSize = blockSize;
    config.overlap = overlap;
    config.stepX = blockSize - overlap;
    config.stepY = blockSize - overlap;
    config.pel = pel;
    config.delta = delta;
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
        analysisData.nDeltaFrame = config.delta;
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

static ResolvedRIFEModel resolveRIFEModel(std::string modelPath) {
    ResolvedRIFEModel resolved{};
    resolved.modelPath = std::move(modelPath);
    resolved.padding = 32;

    if (resolved.modelPath.empty())
        throw "model_path must be specified";

    std::ifstream ifs{ resolved.modelPath + "/flownet.param" };
    if (!ifs.is_open())
        throw "failed to load model";

    if (resolved.modelPath.find("rife-v2") != std::string::npos)
        resolved.rifeV2 = true;
    else if (resolved.modelPath.find("rife-v3.9") != std::string::npos)
        resolved.rifeV4 = true;
    else if (resolved.modelPath.find("rife-v3") != std::string::npos)
        resolved.rifeV2 = true;
    else if (resolved.modelPath.find("rife-v4") != std::string::npos)
        resolved.rifeV4 = true;
    else if (resolved.modelPath.find("rife4") != std::string::npos)
        resolved.rifeV4 = true;

    if (resolved.modelPath.find("rife-v4.25") != std::string::npos)
        resolved.padding = 64;
    if (resolved.modelPath.find("rife-v4.25-lite") != std::string::npos)
        resolved.padding = 128;
    if (resolved.modelPath.find("rife-v4.26") != std::string::npos)
        resolved.padding = 64;
    else if (resolved.modelPath.find("rife") == std::string::npos)
        throw "unknown model dir type";

    return resolved;
}

static bool isEarlyUnsupportedRIFEV4Model(const std::string& modelPath) {
    const auto plainRifeV4Path = modelPath.find("rife-v4") != std::string::npos &&
                                 modelPath.find("rife-v4.") == std::string::npos;
    if (plainRifeV4Path)
        return true;

    return modelPath.find("rife-v4.0") != std::string::npos ||
           modelPath.find("rife-v4.1") != std::string::npos ||
           modelPath.find("rife4.0") != std::string::npos ||
           modelPath.find("rife4.1") != std::string::npos;
}

static bool supportsMotionVectorExport(const ResolvedRIFEModel& resolvedModel) {
    if (resolvedModel.modelPath.find("rife-v3.1") != std::string::npos)
        return true;

    const auto isV4FamilyPath = resolvedModel.modelPath.find("rife-v4") != std::string::npos ||
                                resolvedModel.modelPath.find("rife4") != std::string::npos;
    if (!isV4FamilyPath)
        return false;

    return !isEarlyUnsupportedRIFEV4Model(resolvedModel.modelPath);
}

static void loadRIFEModel(RIFE& rife, const std::string& modelPath) {
#ifdef _WIN32
    const auto bufferSize = MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wbuffer(bufferSize);
    MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, wbuffer.data(), bufferSize);
    rife.load(wbuffer.data());
#else
    rife.load(modelPath);
#endif
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

struct RIFEMVPairData final {
    VSNode* node;
    VSVideoInfo vi;
    MotionVectorConfig mvConfig;
    std::unique_ptr<RIFE> rife;
    std::unique_ptr<std::counting_semaphore<>> semaphore;
};

struct RIFEMVOutputData final {
    VSNode* node;
    VSVideoInfo vi;
    MVAnalysisData analysisData;
    std::vector<char> invalidBlob;
    bool backward;
};

struct RIFEMVApproxPairData final {
    VSNode* node;
    VSVideoInfo vi;
    MotionVectorConfig mvConfig;
    std::unique_ptr<RIFE> rife;
    std::unique_ptr<std::counting_semaphore<>> semaphore;
};

struct RIFEMVApproxOutputData final {
    VSNode* node;
    VSNode* sourceNode;
    VSVideoInfo vi;
    MotionVectorConfig mvConfig;
    MVAnalysisData analysisData;
    std::vector<char> invalidBlob;
    bool backward;
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

struct SADContext final {
    int width;
    int height;
    int stride;
    int blockSize;
    bool useChroma;
    double maxSample;
    const float* currentR;
    const float* currentG;
    const float* currentB;
    const float* referenceR;
    const float* referenceG;
    const float* referenceB;
    const float* currentLuma;
    const float* referenceLuma;
};

static void buildFrameLumaPlane(const VSFrame* frame, const int width, const int height, const int stride,
                                std::vector<float>& luma, const VSAPI* vsapi) noexcept {
    luma.resize(static_cast<size_t>(stride) * height);
    const auto* planeR = reinterpret_cast<const float*>(vsapi->getReadPtr(frame, 0));
    const auto* planeG = reinterpret_cast<const float*>(vsapi->getReadPtr(frame, 1));
    const auto* planeB = reinterpret_cast<const float*>(vsapi->getReadPtr(frame, 2));

    for (auto y = 0; y < height; y++) {
        const auto row = static_cast<size_t>(y) * stride;
        for (auto x = 0; x < width; x++) {
            const auto idx = row + x;
            luma[idx] = static_cast<float>(rgbToLuma(planeR[idx], planeG[idx], planeB[idx]));
        }
    }
}

static SADContext makeSADContext(const VSFrame* current, const VSFrame* reference, const RIFEData* const VS_RESTRICT d,
                                 const VSAPI* vsapi, const float* currentLuma, const float* referenceLuma) noexcept {
    const auto stride = vsapi->getStride(current, 0) / vsapi->getVideoFrameFormat(current)->bytesPerSample;
    SADContext context{};
    context.width = vsapi->getFrameWidth(current, 0);
    context.height = vsapi->getFrameHeight(current, 0);
    context.stride = stride;
    context.blockSize = d->mvBlockSize;
    context.useChroma = d->mvUseChroma;
    context.maxSample = static_cast<double>((1ULL << d->mvBits) - 1ULL);
    context.currentR = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 0));
    context.currentG = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 1));
    context.currentB = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 2));
    context.referenceR = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 0));
    context.referenceG = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 1));
    context.referenceB = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 2));
    context.currentLuma = currentLuma;
    context.referenceLuma = referenceLuma;
    return context;
}

static int64_t computeBlockSAD(const SADContext& context, const int pixelDx, const int pixelDy,
                               const int blockX, const int blockY) noexcept {
    int64_t sad{};
    const auto currentX0 = blockX;
    const auto currentY0 = blockY;
    const auto referenceX0 = blockX + pixelDx;
    const auto referenceY0 = blockY + pixelDy;
    const auto interior = currentX0 >= 0 && currentY0 >= 0 &&
                          referenceX0 >= 0 && referenceY0 >= 0 &&
                          currentX0 + context.blockSize <= context.width &&
                          currentY0 + context.blockSize <= context.height &&
                          referenceX0 + context.blockSize <= context.width &&
                          referenceY0 + context.blockSize <= context.height;

    if (interior) {
        if (context.useChroma) {
            for (auto y = 0; y < context.blockSize; y++) {
                const auto currentRow = (currentY0 + y) * context.stride + currentX0;
                const auto referenceRow = (referenceY0 + y) * context.stride + referenceX0;
                for (auto x = 0; x < context.blockSize; x++) {
                    const auto currentIndex = currentRow + x;
                    const auto referenceIndex = referenceRow + x;
                    sad += static_cast<int64_t>(std::llround((std::abs(context.currentR[currentIndex] - context.referenceR[referenceIndex]) +
                                                              std::abs(context.currentG[currentIndex] - context.referenceG[referenceIndex]) +
                                                              std::abs(context.currentB[currentIndex] - context.referenceB[referenceIndex])) * context.maxSample));
                }
            }
        } else {
            for (auto y = 0; y < context.blockSize; y++) {
                const auto currentRow = (currentY0 + y) * context.stride + currentX0;
                const auto referenceRow = (referenceY0 + y) * context.stride + referenceX0;
                for (auto x = 0; x < context.blockSize; x++) {
                    const auto currentIndex = currentRow + x;
                    const auto referenceIndex = referenceRow + x;
                    sad += static_cast<int64_t>(std::llround(std::abs(context.currentLuma[currentIndex] - context.referenceLuma[referenceIndex]) * context.maxSample));
                }
            }
        }

        return sad;
    }

    for (auto y = 0; y < context.blockSize; y++) {
        const auto currentY = clampPixel(blockY + y, context.height);
        const auto referenceY = clampPixel(currentY + pixelDy, context.height);
        for (auto x = 0; x < context.blockSize; x++) {
            const auto currentX = clampPixel(blockX + x, context.width);
            const auto referenceX = clampPixel(currentX + pixelDx, context.width);
            const auto currentIndex = currentY * context.stride + currentX;
            const auto referenceIndex = referenceY * context.stride + referenceX;

            if (context.useChroma) {
                sad += static_cast<int64_t>(std::llround((std::abs(context.currentR[currentIndex] - context.referenceR[referenceIndex]) +
                                                          std::abs(context.currentG[currentIndex] - context.referenceG[referenceIndex]) +
                                                          std::abs(context.currentB[currentIndex] - context.referenceB[referenceIndex])) * context.maxSample));
            } else {
                sad += static_cast<int64_t>(std::llround(std::abs(context.currentLuma[currentIndex] - context.referenceLuma[referenceIndex]) * context.maxSample));
            }
        }
    }

    return sad;
}

static std::vector<char> packMotionVectorBlob(const std::vector<MVToolsVector>& vectors, const bool valid) {
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

static std::vector<char> buildMVToolsVectorBlob(const VSFrame* current, const VSFrame* reference, const float* flow,
                                                const bool valid, const RIFEData* const VS_RESTRICT d,
                                                const VSAPI* vsapi,
                                                const std::vector<float>* currentLumaCache = nullptr,
                                                const std::vector<float>* referenceLumaCache = nullptr) {
    const auto vectorCount = static_cast<size_t>(d->mvBlkX) * d->mvBlkY;
    std::vector<MVToolsVector> vectors(vectorCount);

    if (!valid) {
        for (auto& vector : vectors) {
            vector.x = 0;
            vector.y = 0;
            vector.sad = d->mvInvalidSad;
        }

        return packMotionVectorBlob(vectors, false);
    }

    const auto width = vsapi->getFrameWidth(current, 0);
    const auto height = vsapi->getFrameHeight(current, 0);
    const auto stride = vsapi->getStride(current, 0) / vsapi->getVideoFrameFormat(current)->bytesPerSample;
    std::vector<float> currentLuma;
    std::vector<float> referenceLuma;
    const float* currentLumaPtr = nullptr;
    const float* referenceLumaPtr = nullptr;
    if (!d->mvUseChroma) {
        if (currentLumaCache && currentLumaCache->size() >= static_cast<size_t>(stride) * height) {
            currentLumaPtr = currentLumaCache->data();
        } else {
            buildFrameLumaPlane(current, width, height, stride, currentLuma, vsapi);
            currentLumaPtr = currentLuma.data();
        }

        if (referenceLumaCache && referenceLumaCache->size() >= static_cast<size_t>(stride) * height) {
            referenceLumaPtr = referenceLumaCache->data();
        } else {
            buildFrameLumaPlane(reference, width, height, stride, referenceLuma, vsapi);
            referenceLumaPtr = referenceLuma.data();
        }
    }

    const auto sadContext = makeSADContext(current, reference, d, vsapi, currentLumaPtr, referenceLumaPtr);
    const auto channelOffset = d->mvBackward ? 0 : 2;
    const auto flowPlaneSize = width * height;
    const auto* flowXPlane = flow + (channelOffset + 0) * flowPlaneSize;
    const auto* flowYPlane = flow + (channelOffset + 1) * flowPlaneSize;

    for (auto by = 0; by < d->mvBlkY; by++) {
        const auto blockY = by * d->mvStepY - d->mvVPadding;
        for (auto bx = 0; bx < d->mvBlkX; bx++) {
            const auto blockX = bx * d->mvStepX - d->mvHPadding;
            auto& vector = vectors[static_cast<size_t>(by) * d->mvBlkX + bx];
            const auto flowX = reduceBlockFlow(flowXPlane, width, height, blockX, blockY, d);
            const auto flowY = reduceBlockFlow(flowYPlane, width, height, blockX, blockY, d);

            vector.x = static_cast<int>(std::lround(-2.0f * flowX * d->mvPel));
            vector.y = static_cast<int>(std::lround(-2.0f * flowY * d->mvPel));
            vector.x = clampMotionVectorComponent(vector.x, d->mvPel, blockX, d->mvBlockSize, width, d->mvHPadding);
            vector.y = clampMotionVectorComponent(vector.y, d->mvPel, blockY, d->mvBlockSize, height, d->mvVPadding);
            const auto pixelDx = static_cast<int>(std::lround(static_cast<double>(vector.x) / d->mvPel));
            const auto pixelDy = static_cast<int>(std::lround(static_cast<double>(vector.y) / d->mvPel));
            vector.sad = computeBlockSAD(sadContext, pixelDx, pixelDy, blockX, blockY);
        }
    }

    return packMotionVectorBlob(vectors, true);
}

static RIFEData makeMotionVectorBuilderData(const MotionVectorConfig& config, const bool backward) {
    RIFEData d{};
    d.mvBackward = backward;
    d.mvUseChroma = config.useChroma;
    d.mvBlockSize = config.blockSize;
    d.mvOverlap = config.overlap;
    d.mvStepX = config.stepX;
    d.mvStepY = config.stepY;
    d.mvPel = config.pel;
    d.mvBits = config.bits;
    d.mvHPadding = config.hPadding;
    d.mvVPadding = config.vPadding;
    d.mvBlkX = config.blkX;
    d.mvBlkY = config.blkY;
    d.mvBlockReduce = config.blockReduce;
    d.mvInvalidSad = config.invalidSad;
    return d;
}

static std::vector<char> buildMotionVectorBlobFromConfig(const VSFrame* current, const VSFrame* reference, const float* flow,
                                                         const bool valid, const MotionVectorConfig& config,
                                                         const bool backward, const VSAPI* vsapi,
                                                         const std::vector<float>* currentLumaCache = nullptr,
                                                         const std::vector<float>* referenceLumaCache = nullptr) {
    const auto d = makeMotionVectorBuilderData(config, backward);
    return buildMVToolsVectorBlob(current, reference, flow, valid, &d, vsapi, currentLumaCache, referenceLumaCache);
}

static std::vector<char> buildInvalidMotionVectorBlob(const MotionVectorConfig& config, const bool backward) {
    return buildMotionVectorBlobFromConfig(nullptr, nullptr, nullptr, false, config, backward, nullptr);
}

static float sampleBilinearPlane(const float* data, const int width, const int height, float x, float y) noexcept {
    x = std::clamp(x, 0.0f, static_cast<float>(width - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(height - 1));

    const auto x0 = static_cast<int>(std::floor(x));
    const auto y0 = static_cast<int>(std::floor(y));
    const auto x1 = std::min(x0 + 1, width - 1);
    const auto y1 = std::min(y0 + 1, height - 1);
    const auto alpha = x - x0;
    const auto beta = y - y0;
    const auto row0 = static_cast<size_t>(y0) * width;
    const auto row1 = static_cast<size_t>(y1) * width;
    const auto top = data[row0 + x0] * (1.0f - alpha) + data[row0 + x1] * alpha;
    const auto bottom = data[row1 + x0] * (1.0f - alpha) + data[row1 + x1] * alpha;

    return top * (1.0f - beta) + bottom * beta;
}

static std::vector<float> buildDisplacementFromFlow(const float* flow, const int width, const int height,
                                                    const int channelOffset) {
    const auto planeSize = static_cast<size_t>(width) * height;
    std::vector<float> displacement(planeSize * 2);

    for (size_t i = 0; i < planeSize; i++) {
        displacement[i] = -2.0f * flow[(static_cast<size_t>(channelOffset) + 0) * planeSize + i];
        displacement[planeSize + i] = -2.0f * flow[(static_cast<size_t>(channelOffset) + 1) * planeSize + i];
    }

    return displacement;
}

static bool getDisplacementPlanes(const VSFrame* frame, const char* key, const int width, const int height,
                                  const float*& displacementX, const float*& displacementY,
                                  const VSAPI* vsapi) noexcept {
    const auto props = vsapi->getFramePropertiesRO(frame);
    int err{};
    const auto* data = vsapi->mapGetData(props, key, 0, &err);
    if (err)
        return false;

    const auto expectedSize = static_cast<int>(sizeof(float) * static_cast<size_t>(width) * height * 2);
    if (vsapi->mapGetDataSize(props, key, 0, nullptr) != expectedSize)
        return false;

    displacementX = reinterpret_cast<const float*>(data);
    displacementY = displacementX + static_cast<size_t>(width) * height;
    return true;
}

static void composeDisplacementSequence(const std::vector<const float*>& displacementXs,
                                        const std::vector<const float*>& displacementYs,
                                        const int width, const int height,
                                        std::vector<float>& composedX,
                                        std::vector<float>& composedY) {
    const auto planeSize = static_cast<size_t>(width) * height;
    composedX.assign(displacementXs.front(), displacementXs.front() + planeSize);
    composedY.assign(displacementYs.front(), displacementYs.front() + planeSize);

    for (size_t sequenceIndex = 1; sequenceIndex < displacementXs.size(); sequenceIndex++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                const auto index = static_cast<size_t>(y) * width + x;
                const auto sampleX = static_cast<float>(x) + composedX[index];
                const auto sampleY = static_cast<float>(y) + composedY[index];
                composedX[index] += sampleBilinearPlane(displacementXs[sequenceIndex], width, height, sampleX, sampleY);
                composedY[index] += sampleBilinearPlane(displacementYs[sequenceIndex], width, height, sampleX, sampleY);
            }
        }
    }
}

static std::vector<char> buildMotionVectorBlobFromDisplacement(const VSFrame* current, const VSFrame* reference,
                                                               const float* displacementX, const float* displacementY,
                                                               const bool valid, const MotionVectorConfig& config,
                                                               const bool backward, const VSAPI* vsapi,
                                                               const std::vector<float>* currentLumaCache = nullptr,
                                                               const std::vector<float>* referenceLumaCache = nullptr) {
    const auto d = makeMotionVectorBuilderData(config, backward);
    const auto vectorCount = static_cast<size_t>(d.mvBlkX) * d.mvBlkY;
    std::vector<MVToolsVector> vectors(vectorCount);

    if (!valid) {
        for (auto& vector : vectors) {
            vector.x = 0;
            vector.y = 0;
            vector.sad = d.mvInvalidSad;
        }
        return packMotionVectorBlob(vectors, false);
    }

    const auto width = vsapi->getFrameWidth(current, 0);
    const auto height = vsapi->getFrameHeight(current, 0);
    const auto stride = vsapi->getStride(current, 0) / vsapi->getVideoFrameFormat(current)->bytesPerSample;
    std::vector<float> currentLuma;
    std::vector<float> referenceLuma;
    const float* currentLumaPtr = nullptr;
    const float* referenceLumaPtr = nullptr;
    if (!d.mvUseChroma) {
        if (currentLumaCache && currentLumaCache->size() >= static_cast<size_t>(stride) * height) {
            currentLumaPtr = currentLumaCache->data();
        } else {
            buildFrameLumaPlane(current, width, height, stride, currentLuma, vsapi);
            currentLumaPtr = currentLuma.data();
        }

        if (referenceLumaCache && referenceLumaCache->size() >= static_cast<size_t>(stride) * height) {
            referenceLumaPtr = referenceLumaCache->data();
        } else {
            buildFrameLumaPlane(reference, width, height, stride, referenceLuma, vsapi);
            referenceLumaPtr = referenceLuma.data();
        }
    }

    const auto sadContext = makeSADContext(current, reference, &d, vsapi, currentLumaPtr, referenceLumaPtr);
    for (auto by = 0; by < d.mvBlkY; by++) {
        const auto blockY = by * d.mvStepY - d.mvVPadding;
        for (auto bx = 0; bx < d.mvBlkX; bx++) {
            const auto blockX = bx * d.mvStepX - d.mvHPadding;
            auto& vector = vectors[static_cast<size_t>(by) * d.mvBlkX + bx];
            const auto pixelDx = reduceBlockFlow(displacementX, width, height, blockX, blockY, &d);
            const auto pixelDy = reduceBlockFlow(displacementY, width, height, blockX, blockY, &d);

            vector.x = static_cast<int>(std::lround(pixelDx * d.mvPel));
            vector.y = static_cast<int>(std::lround(pixelDy * d.mvPel));
            vector.x = clampMotionVectorComponent(vector.x, d.mvPel, blockX, d.mvBlockSize, width, d.mvHPadding);
            vector.y = clampMotionVectorComponent(vector.y, d.mvPel, blockY, d.mvBlockSize, height, d.mvVPadding);
            const auto vectorPixelDx = static_cast<int>(std::lround(static_cast<double>(vector.x) / d.mvPel));
            const auto vectorPixelDy = static_cast<int>(std::lround(static_cast<double>(vector.y) / d.mvPel));
            vector.sad = computeBlockSAD(sadContext, vectorPixelDx, vectorPixelDy, blockX, blockY);
        }
    }

    return packMotionVectorBlob(vectors, true);
}

static void zeroMotionVectorFrame(VSFrame* frame, const VSVideoInfo& vi, const VSAPI* vsapi);

static VSFrame* createMotionVectorFrame(const VSVideoInfo& vi, const MVAnalysisData& analysisData,
                                        const char* vectorBlob, const int vectorBlobSize,
                                        VSCore* core, const VSAPI* vsapi) {
    auto dst = vsapi->newVideoFrame(&vi.format, vi.width, vi.height, nullptr, core);
    zeroMotionVectorFrame(dst, vi, vsapi);
    auto props = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetData(props, MVToolsAnalysisDataKey, reinterpret_cast<const char*>(&analysisData), sizeof(analysisData), dtBinary, maReplace);
    vsapi->mapSetData(props, MVToolsVectorsKey, vectorBlob, vectorBlobSize, dtBinary, maReplace);
    return dst;
}

static void zeroMotionVectorFrame(VSFrame* frame, const VSVideoInfo& vi, const VSAPI* vsapi) {
    auto* dstp = vsapi->getWritePtr(frame, 0);
    const auto dstStride = vsapi->getStride(frame, 0);
    for (auto y = 0; y < vi.height; y++)
        std::memset(dstp + static_cast<size_t>(y) * dstStride, 0, vi.width * vi.format.bytesPerSample);
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
        const auto delta = d->mvConfig.delta;
        if (activationReason == arInitial) {
            vsapi->requestFrameFilter(n, d->node, frameCtx);
            if (d->mvBackward) {
                if (n + delta < d->vi.numFrames)
                    vsapi->requestFrameFilter(n + delta, d->node, frameCtx);
            } else if (n >= delta) {
                vsapi->requestFrameFilter(n - delta, d->node, frameCtx);
            }
        } else if (activationReason == arAllFramesReady) {
            auto current = vsapi->getFrameFilter(n, d->node, frameCtx);
            const VSFrame* reference{};
            if (d->mvBackward) {
                if (n + delta < d->vi.numFrames)
                    reference = vsapi->getFrameFilter(n + delta, d->node, frameCtx);
            } else if (n >= delta) {
                reference = vsapi->getFrameFilter(n - delta, d->node, frameCtx);
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

static const VSFrame* VS_CC rifeMVPairGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData,
                                               VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEMVPairData*>(instanceData) };
    const auto delta = d->mvConfig.delta;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (n + delta < d->vi.numFrames)
            vsapi->requestFrameFilter(n + delta, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto current = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrame* reference = n + delta < d->vi.numFrames ? vsapi->getFrameFilter(n + delta, d->node, frameCtx) : nullptr;

        auto dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, current, core);
        zeroMotionVectorFrame(dst, d->vi, vsapi);
        auto props = vsapi->getFramePropertiesRW(dst);

        std::vector<char> backwardBlob;
        std::vector<char> forwardBlob;
        if (reference) {
            const auto width = vsapi->getFrameWidth(current, 0);
            const auto height = vsapi->getFrameHeight(current, 0);
            const auto stride = vsapi->getStride(current, 0) / vsapi->getVideoFrameFormat(current)->bytesPerSample;
            std::vector<float> flow(static_cast<size_t>(width) * height * 4);
            std::vector<float> currentLuma;
            std::vector<float> referenceLuma;
            const auto currentR = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 0));
            const auto currentG = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 1));
            const auto currentB = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 2));
            const auto referenceR = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 0));
            const auto referenceG = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 1));
            const auto referenceB = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 2));

            d->semaphore->acquire();
            const auto status = d->rife->process_flow(currentR, currentG, currentB, referenceR, referenceG, referenceB, flow.data(), width, height, stride);
            d->semaphore->release();
            if (status != 0) {
                vsapi->freeFrame(current);
                vsapi->freeFrame(reference);
                vsapi->freeFrame(dst);
                vsapi->setFilterError("RIFEMV: failed to export motion vectors", frameCtx);
                return nullptr;
            }

            if (!d->mvConfig.useChroma) {
                buildFrameLumaPlane(current, width, height, stride, currentLuma, vsapi);
                buildFrameLumaPlane(reference, width, height, stride, referenceLuma, vsapi);
            }

            backwardBlob = buildMotionVectorBlobFromConfig(current, reference, flow.data(), true, d->mvConfig, true, vsapi,
                                                           currentLuma.empty() ? nullptr : &currentLuma,
                                                           referenceLuma.empty() ? nullptr : &referenceLuma);
            forwardBlob = buildMotionVectorBlobFromConfig(reference, current, flow.data(), true, d->mvConfig, false, vsapi,
                                                          referenceLuma.empty() ? nullptr : &referenceLuma,
                                                          currentLuma.empty() ? nullptr : &currentLuma);
        } else {
            backwardBlob = buildInvalidMotionVectorBlob(d->mvConfig, true);
            forwardBlob = buildInvalidMotionVectorBlob(d->mvConfig, false);
        }

        vsapi->mapSetData(props, RIFEMVBackwardVectorsInternalKey, backwardBlob.data(), static_cast<int>(backwardBlob.size()), dtBinary, maReplace);
        vsapi->mapSetData(props, RIFEMVForwardVectorsInternalKey, forwardBlob.data(), static_cast<int>(forwardBlob.size()), dtBinary, maReplace);

        vsapi->freeFrame(current);
        vsapi->freeFrame(reference);
        return dst;
    }

    return nullptr;
}

static const VSFrame* VS_CC rifeMVOutputGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData,
                                                 VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEMVOutputData*>(instanceData) };
    const auto delta = d->analysisData.nDeltaFrame;
    const auto pairIndex = d->backward ? n : n - delta;

    if (activationReason == arInitial) {
        if (pairIndex >= 0 && pairIndex < d->vi.numFrames) {
            vsapi->requestFrameFilter(pairIndex, d->node, frameCtx);
        } else {
            auto dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, nullptr, core);
            zeroMotionVectorFrame(dst, d->vi, vsapi);
            auto props = vsapi->getFramePropertiesRW(dst);
            vsapi->mapSetData(props, MVToolsAnalysisDataKey, reinterpret_cast<const char*>(&d->analysisData), sizeof(d->analysisData), dtBinary, maReplace);
            vsapi->mapSetData(props, MVToolsVectorsKey, d->invalidBlob.data(), static_cast<int>(d->invalidBlob.size()), dtBinary, maReplace);
            return dst;
        }
    } else if (activationReason == arAllFramesReady) {
        const VSFrame* pairFrame{};
        if (pairIndex >= 0 && pairIndex < d->vi.numFrames)
            pairFrame = vsapi->getFrameFilter(pairIndex, d->node, frameCtx);

        VSFrame* dst{};
        const char* vectorBlob = nullptr;
        int vectorBlobSize{};
        if (pairFrame) {
            const auto pairProps = vsapi->getFramePropertiesRO(pairFrame);
            const auto blobKey = d->backward ? RIFEMVBackwardVectorsInternalKey : RIFEMVForwardVectorsInternalKey;
            vectorBlob = vsapi->mapGetData(pairProps, blobKey, 0, nullptr);
            vectorBlobSize = vsapi->mapGetDataSize(pairProps, blobKey, 0, nullptr);
            dst = vsapi->copyFrame(pairFrame, core);
        } else {
            dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, nullptr, core);
            zeroMotionVectorFrame(dst, d->vi, vsapi);
            vectorBlob = d->invalidBlob.data();
            vectorBlobSize = static_cast<int>(d->invalidBlob.size());
        }

        auto props = vsapi->getFramePropertiesRW(dst);
        vsapi->mapDeleteKey(props, RIFEMVBackwardVectorsInternalKey);
        vsapi->mapDeleteKey(props, RIFEMVForwardVectorsInternalKey);
        vsapi->mapSetData(props, MVToolsAnalysisDataKey, reinterpret_cast<const char*>(&d->analysisData), sizeof(d->analysisData), dtBinary, maReplace);
        vsapi->mapSetData(props, MVToolsVectorsKey, vectorBlob, vectorBlobSize, dtBinary, maReplace);

        vsapi->freeFrame(pairFrame);
        return dst;
    }

    return nullptr;
}

static const VSFrame* VS_CC rifeMVApproxPairGetFrame(int n, int activationReason, void* instanceData,
                                                     [[maybe_unused]] void** frameData,
                                                     VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEMVApproxPairData*>(instanceData) };

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (n + 1 < d->vi.numFrames)
            vsapi->requestFrameFilter(n + 1, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto current = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrame* reference = n + 1 < d->vi.numFrames ? vsapi->getFrameFilter(n + 1, d->node, frameCtx) : nullptr;

        auto dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, current, core);
        zeroMotionVectorFrame(dst, d->vi, vsapi);
        auto props = vsapi->getFramePropertiesRW(dst);

        std::vector<char> backwardBlob;
        std::vector<char> forwardBlob;
        std::vector<float> backwardDisplacement;
        std::vector<float> forwardDisplacement;
        const auto planeSize = static_cast<size_t>(d->vi.width) * d->vi.height;

        if (reference) {
            const auto width = vsapi->getFrameWidth(current, 0);
            const auto height = vsapi->getFrameHeight(current, 0);
            const auto stride = vsapi->getStride(current, 0) / vsapi->getVideoFrameFormat(current)->bytesPerSample;
            std::vector<float> flow(static_cast<size_t>(width) * height * 4);
            std::vector<float> currentLuma;
            std::vector<float> referenceLuma;
            const auto currentR = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 0));
            const auto currentG = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 1));
            const auto currentB = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 2));
            const auto referenceR = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 0));
            const auto referenceG = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 1));
            const auto referenceB = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 2));

            d->semaphore->acquire();
            const auto status = d->rife->process_flow(currentR, currentG, currentB, referenceR, referenceG, referenceB,
                                                      flow.data(), width, height, stride);
            d->semaphore->release();
            if (status != 0) {
                vsapi->freeFrame(current);
                vsapi->freeFrame(reference);
                vsapi->freeFrame(dst);
                vsapi->setFilterError("RIFEMVApprox: failed to export motion vectors", frameCtx);
                return nullptr;
            }

            if (!d->mvConfig.useChroma) {
                buildFrameLumaPlane(current, width, height, stride, currentLuma, vsapi);
                buildFrameLumaPlane(reference, width, height, stride, referenceLuma, vsapi);
            }

            backwardBlob = buildMotionVectorBlobFromConfig(current, reference, flow.data(), true, d->mvConfig, true, vsapi,
                                                           currentLuma.empty() ? nullptr : &currentLuma,
                                                           referenceLuma.empty() ? nullptr : &referenceLuma);
            forwardBlob = buildMotionVectorBlobFromConfig(reference, current, flow.data(), true, d->mvConfig, false, vsapi,
                                                          referenceLuma.empty() ? nullptr : &referenceLuma,
                                                          currentLuma.empty() ? nullptr : &currentLuma);
            backwardDisplacement = buildDisplacementFromFlow(flow.data(), width, height, 0);
            forwardDisplacement = buildDisplacementFromFlow(flow.data(), width, height, 2);
        } else {
            backwardBlob = buildInvalidMotionVectorBlob(d->mvConfig, true);
            forwardBlob = buildInvalidMotionVectorBlob(d->mvConfig, false);
            backwardDisplacement.assign(planeSize * 2, 0.0f);
            forwardDisplacement.assign(planeSize * 2, 0.0f);
        }

        vsapi->mapSetData(props, RIFEMVBackwardVectorsInternalKey, backwardBlob.data(), static_cast<int>(backwardBlob.size()), dtBinary, maReplace);
        vsapi->mapSetData(props, RIFEMVForwardVectorsInternalKey, forwardBlob.data(), static_cast<int>(forwardBlob.size()), dtBinary, maReplace);
        vsapi->mapSetData(props, RIFEMVBackwardDisplacementInternalKey,
                          reinterpret_cast<const char*>(backwardDisplacement.data()),
                          static_cast<int>(backwardDisplacement.size() * sizeof(float)), dtBinary, maReplace);
        vsapi->mapSetData(props, RIFEMVForwardDisplacementInternalKey,
                          reinterpret_cast<const char*>(forwardDisplacement.data()),
                          static_cast<int>(forwardDisplacement.size() * sizeof(float)), dtBinary, maReplace);

        vsapi->freeFrame(current);
        vsapi->freeFrame(reference);
        return dst;
    }

    return nullptr;
}

static const VSFrame* VS_CC rifeMVApproxOutputGetFrame(int n, int activationReason, void* instanceData,
                                                       [[maybe_unused]] void** frameData,
                                                       VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEMVApproxOutputData*>(instanceData) };
    const auto delta = d->analysisData.nDeltaFrame;
    const auto valid = d->backward ? n + delta < d->vi.numFrames : n >= delta;
    const auto createInvalidFrame = [&]() {
        return createMotionVectorFrame(d->vi, d->analysisData, d->invalidBlob.data(), static_cast<int>(d->invalidBlob.size()), core, vsapi);
    };

    if (activationReason == arInitial) {
        if (!valid)
            return createInvalidFrame();

        for (auto i = 0; i < delta; i++) {
            const auto pairIndex = d->backward ? n + i : n - 1 - i;
            vsapi->requestFrameFilter(pairIndex, d->node, frameCtx);
        }

        if (delta > 1) {
            vsapi->requestFrameFilter(n, d->sourceNode, frameCtx);
            vsapi->requestFrameFilter(d->backward ? n + delta : n - delta, d->sourceNode, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        if (!valid)
            return createInvalidFrame();

        std::vector<const VSFrame*> pairFrames(delta);
        const VSFrame* current{};
        const VSFrame* reference{};
        const auto cleanup = [&]() {
            for (const auto* pairFrame : pairFrames) {
                if (pairFrame)
                    vsapi->freeFrame(pairFrame);
            }
            if (current)
                vsapi->freeFrame(current);
            if (reference)
                vsapi->freeFrame(reference);
        };

        for (auto i = 0; i < delta; i++) {
            const auto pairIndex = d->backward ? n + i : n - 1 - i;
            pairFrames[i] = vsapi->getFrameFilter(pairIndex, d->node, frameCtx);
        }

        if (delta == 1) {
            const auto props = vsapi->getFramePropertiesRO(pairFrames[0]);
            const auto blobKey = d->backward ? RIFEMVBackwardVectorsInternalKey : RIFEMVForwardVectorsInternalKey;
            int err{};
            const auto* vectorBlob = vsapi->mapGetData(props, blobKey, 0, &err);
            const auto vectorBlobSize = err ? 0 : vsapi->mapGetDataSize(props, blobKey, 0, nullptr);
            if (err || !vectorBlob || vectorBlobSize <= 0) {
                cleanup();
                vsapi->setFilterError("RIFEMVApprox: missing internal vector data", frameCtx);
                return nullptr;
            }

            auto dst = createMotionVectorFrame(d->vi, d->analysisData, vectorBlob, vectorBlobSize, core, vsapi);
            cleanup();
            return dst;
        }

        current = vsapi->getFrameFilter(n, d->sourceNode, frameCtx);
        reference = vsapi->getFrameFilter(d->backward ? n + delta : n - delta, d->sourceNode, frameCtx);
        const auto width = vsapi->getFrameWidth(current, 0);
        const auto height = vsapi->getFrameHeight(current, 0);
        const auto displacementKey = d->backward ? RIFEMVBackwardDisplacementInternalKey : RIFEMVForwardDisplacementInternalKey;
        std::vector<const float*> displacementXs(delta);
        std::vector<const float*> displacementYs(delta);

        for (auto i = 0; i < delta; i++) {
            if (!getDisplacementPlanes(pairFrames[i], displacementKey, width, height,
                                       displacementXs[i], displacementYs[i], vsapi)) {
                cleanup();
                vsapi->setFilterError("RIFEMVApprox: missing internal displacement data", frameCtx);
                return nullptr;
            }
        }

        std::vector<float> composedX;
        std::vector<float> composedY;
        composeDisplacementSequence(displacementXs, displacementYs, width, height, composedX, composedY);
        const auto vectorBlob = buildMotionVectorBlobFromDisplacement(current, reference,
                                                                      composedX.data(), composedY.data(), true,
                                                                      d->mvConfig, d->backward, vsapi);

        auto dst = createMotionVectorFrame(d->vi, d->analysisData, vectorBlob.data(), static_cast<int>(vectorBlob.size()), core, vsapi);
        cleanup();
        return dst;
    }

    return nullptr;
}

static void VS_CC rifeMVApproxPairFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<RIFEMVApproxPairData*>(instanceData) };
    vsapi->freeNode(d->node);
    delete d;

    if (--numGPUInstances == 0)
        ncnn::destroy_gpu_instance();
}

static void VS_CC rifeMVApproxOutputFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<RIFEMVApproxOutputData*>(instanceData) };
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->sourceNode);
    delete d;
}

static void VS_CC rifeMVPairFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<RIFEMVPairData*>(instanceData) };
    vsapi->freeNode(d->node);
    delete d;

    if (--numGPUInstances == 0)
        ncnn::destroy_gpu_instance();
}

static void VS_CC rifeMVOutputFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<RIFEMVOutputData*>(instanceData) };
    vsapi->freeNode(d->node);
    delete d;
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

        auto flowScale{ static_cast<float>(vsapi->mapGetFloat(in, "flow_scale", 0, &err)) };
        if (err)
            flowScale = 1.f;
        FlowResizeMode flowResizeMode{ FlowResizeMode::Auto };
        const auto cpuFlowResize{ vsapi->mapGetIntSaturated(in, "cpu_flow_resize", 0, &err) };
        if (!err)
            flowResizeMode = cpuFlowResize ? FlowResizeMode::ForceCPU : FlowResizeMode::ForceGPU;
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
        auto mvDelta{ vsapi->mapGetIntSaturated(in, "mv_delta", 0, &err) };
        if (err)
            mvDelta = 1;
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

        if (!std::isfinite(flowScale) || flowScale <= 0.f)
            throw "flow_scale must be greater than 0";

        
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

        const auto resolvedModel = resolveRIFEModel(modelPath);

        if (!d->exportMotionVectors && !resolvedModel.rifeV4 && (d->factorNum != 2 || d->factorDen != 1))
            throw "only rife-v4 model supports custom frame rate";

        if (d->exportMotionVectors) {
            if (d->sceneChange || d->skip)
                throw "mv=True does not support sc or skip";

            if (isEarlyUnsupportedRIFEV4Model(resolvedModel.modelPath))
                throw RIFEMVUnsupportedEarlyV4Error;

            if (!supportsMotionVectorExport(resolvedModel))
                throw RIFEMVModelRequirementError;

            if (mvBlockSize < 1)
                throw "mv_block_size must be at least 1";

            if (mvOverlap < 0 || mvOverlap >= mvBlockSize)
                throw "mv_overlap must be between 0 and mv_block_size - 1";

            if (mvPel < 1)
                throw "mv_pel must be at least 1";

            if (mvDelta < 1)
                throw "mv_delta must be at least 1";

            if (mvBits < 1 || mvBits > 16)
                throw "mv_bits must be between 1 and 16 (inclusive)";

            if (mvHPadding < 0 || mvVPadding < 0)
                throw "mv_hpad and mv_vpad must be non-negative";

            if (mvBlockReduce != MVBlockReduceCenter && mvBlockReduce != MVBlockReduceAverage)
                throw "mv_block_reduce must be 0 (center) or 1 (average)";

            d->mvConfig = createMotionVectorConfig(d->vi, hasMVClip ? &mvClipVi : nullptr,
                                                   d->mvUseChroma, mvBlockSize, mvOverlap,
                                                   mvPel, mvDelta, mvBits, mvHPadding,
                                                   mvVPadding, mvBlockReduce);
            d->mvBlockSize = d->mvConfig.blockSize;
            d->mvOverlap = d->mvConfig.overlap;
            d->mvStepX = d->mvConfig.stepX;
            d->mvStepY = d->mvConfig.stepY;
            d->mvPel = d->mvConfig.pel;
            d->mvBits = d->mvConfig.bits;
            d->mvHPadding = d->mvConfig.hPadding;
            d->mvVPadding = d->mvConfig.vPadding;
            d->mvBlkX = d->mvConfig.blkX;
            d->mvBlkY = d->mvConfig.blkY;
            d->mvBlockReduce = d->mvConfig.blockReduce;
            d->mvInvalidSad = d->mvConfig.invalidSad;
            d->mvAnalysisData = d->mvBackward ? d->mvConfig.backwardAnalysisData : d->mvConfig.forwardAnalysisData;

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

        d->rife = std::make_unique<RIFE>(gpuId, flowScale, 1, resolvedModel.rifeV2, resolvedModel.rifeV4, resolvedModel.padding, flowResizeMode);
        loadRIFEModel(*d->rife, resolvedModel.modelPath);
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
    auto pairData{ std::make_unique<RIFEMVPairData>() };
    VSNode* mvClip{};
    VSNode* pairNode{};
    VSNode* backwardNode{};
    VSNode* forwardNode{};
    bool hasGPUInstance{};

    try {
        pairData->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        pairData->vi = *vsapi->getVideoInfo(pairData->node);
        VSVideoInfo mvClipVi{};
        bool hasMVClip{};
        int err;

        if (!vsh::isConstantVideoFormat(&pairData->vi) ||
            pairData->vi.format.colorFamily != cfRGB ||
            pairData->vi.format.sampleType != stFloat ||
            pairData->vi.format.bitsPerSample != 32)
            throw "only constant RGB format 32 bit float input supported";

        if (ncnn::create_gpu_instance())
            throw "failed to create GPU instance";
        ++numGPUInstances;
        hasGPUInstance = true;

        auto model_path{ vsapi->mapGetData(in, "model_path", 0, &err) };
        std::string modelPath{ err ? "" : model_path };

        auto gpuId{ vsapi->mapGetIntSaturated(in, "gpu_id", 0, &err) };
        if (err)
            gpuId = ncnn::get_default_gpu_index();

        auto gpuThread{ vsapi->mapGetIntSaturated(in, "gpu_thread", 0, &err) };
        if (err)
            gpuThread = 2;

        auto flowScale{ static_cast<float>(vsapi->mapGetFloat(in, "flow_scale", 0, &err)) };
        if (err)
            flowScale = 1.f;
        FlowResizeMode flowResizeMode{ FlowResizeMode::Auto };
        const auto cpuFlowResize{ vsapi->mapGetIntSaturated(in, "cpu_flow_resize", 0, &err) };
        if (!err)
            flowResizeMode = cpuFlowResize ? FlowResizeMode::ForceCPU : FlowResizeMode::ForceGPU;
        auto mvBlockSize{ vsapi->mapGetIntSaturated(in, "mv_block_size", 0, &err) };
        if (err)
            mvBlockSize = 16;
        auto mvOverlap{ vsapi->mapGetIntSaturated(in, "mv_overlap", 0, &err) };
        if (err)
            mvOverlap = 8;
        auto mvPel{ vsapi->mapGetIntSaturated(in, "mv_pel", 0, &err) };
        if (err)
            mvPel = 1;
        auto mvDelta{ vsapi->mapGetIntSaturated(in, "mv_delta", 0, &err) };
        if (err)
            mvDelta = 1;
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
        const auto mvUseChroma = !!vsapi->mapGetInt(in, "mv_chroma", 0, &err);

        mvClip = vsapi->mapGetNode(in, "mv_clip", 0, &err);
        if (!err) {
            mvClipVi = *vsapi->getVideoInfo(mvClip);
            hasMVClip = true;
        }

        if (hasMVClip) {
            if (!vsh::isConstantVideoFormat(&mvClipVi))
                throw "mv_clip must have a constant format";

            if (mvClipVi.width != pairData->vi.width || mvClipVi.height != pairData->vi.height)
                throw "mv_clip dimensions must match clip";

            if (!mvBitsSpecified)
                mvBits = mvClipVi.format.bitsPerSample;
        }

        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count())
            throw "invalid GPU device";

        if (auto queueCount{ ncnn::get_gpu_info(gpuId).compute_queue_count() }; static_cast<uint32_t>(gpuThread) > queueCount)
            std::cerr << "Warning: gpu_thread is recommended to be between 1 and " << queueCount << " (inclusive)" << std::endl;

        if (gpuThread < 1)
            throw "gpu_thread must be greater than 0";

        if (!std::isfinite(flowScale) || flowScale <= 0.f)
            throw "flow_scale must be greater than 0";

        const auto resolvedModel = resolveRIFEModel(modelPath);
        if (isEarlyUnsupportedRIFEV4Model(resolvedModel.modelPath))
            throw RIFEMVUnsupportedEarlyV4Error;

        if (!supportsMotionVectorExport(resolvedModel))
            throw RIFEMVModelRequirementError;

        if (mvBlockSize < 1)
            throw "mv_block_size must be at least 1";

        if (mvOverlap < 0 || mvOverlap >= mvBlockSize)
            throw "mv_overlap must be between 0 and mv_block_size - 1";

        if (mvPel < 1)
            throw "mv_pel must be at least 1";

        if (mvDelta < 1)
            throw "mv_delta must be at least 1";

        if (mvBits < 1 || mvBits > 16)
            throw "mv_bits must be between 1 and 16 (inclusive)";

        if (mvHPadding < 0 || mvVPadding < 0)
            throw "mv_hpad and mv_vpad must be non-negative";

        if (mvBlockReduce != MVBlockReduceCenter && mvBlockReduce != MVBlockReduceAverage)
            throw "mv_block_reduce must be 0 (center) or 1 (average)";

        pairData->mvConfig = createMotionVectorConfig(pairData->vi, hasMVClip ? &mvClipVi : nullptr,
                                                      mvUseChroma, mvBlockSize, mvOverlap, mvPel, mvDelta,
                                                      mvBits, mvHPadding, mvVPadding, mvBlockReduce);

        if (!vsapi->getVideoFormatByID(&pairData->vi.format, pfGray8, core))
            throw "failed to create RIFEMV output format";

        if (mvClip) {
            vsapi->freeNode(mvClip);
            mvClip = nullptr;
        }

        pairData->semaphore = std::make_unique<std::counting_semaphore<>>(gpuThread);
        pairData->rife = std::make_unique<RIFE>(gpuId, flowScale, 1, resolvedModel.rifeV2, resolvedModel.rifeV4, resolvedModel.padding, flowResizeMode);
        loadRIFEModel(*pairData->rife, resolvedModel.modelPath);
    } catch (const char* error) {
        vsapi->mapSetError(out, ("RIFEMV: "s + error).c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(mvClip);

        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }

    const auto outputVi = pairData->vi;
    const auto mvConfig = pairData->mvConfig;
    VSFilterDependency pairDeps[]{ { pairData->node, rpGeneral } };
    pairNode = vsapi->createVideoFilter2("RIFEMVPair", &pairData->vi, rifeMVPairGetFrame, rifeMVPairFree, fmParallel,
                                         pairDeps, 1, pairData.get(), core);
    if (!pairNode) {
        vsapi->mapSetError(out, "RIFEMV: failed to create internal pair filter");
        vsapi->freeNode(pairData->node);
        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }
    pairData.release();

    auto backwardData = std::make_unique<RIFEMVOutputData>();
    backwardData->node = vsapi->addNodeRef(pairNode);
    backwardData->vi = outputVi;
    backwardData->analysisData = mvConfig.backwardAnalysisData;
    backwardData->invalidBlob = buildInvalidMotionVectorBlob(mvConfig, true);
    backwardData->backward = true;
    VSFilterDependency backwardDeps[]{ { backwardData->node, rpGeneral } };
    backwardNode = vsapi->createVideoFilter2("RIFEMVBackward", &backwardData->vi, rifeMVOutputGetFrame, rifeMVOutputFree,
                                             fmParallel, backwardDeps, 1, backwardData.get(), core);
    if (!backwardNode) {
        vsapi->mapSetError(out, "RIFEMV: failed to create backward output filter");
        vsapi->freeNode(backwardData->node);
        vsapi->freeNode(pairNode);
        return;
    }
    backwardData.release();

    auto forwardData = std::make_unique<RIFEMVOutputData>();
    forwardData->node = vsapi->addNodeRef(pairNode);
    forwardData->vi = outputVi;
    forwardData->analysisData = mvConfig.forwardAnalysisData;
    forwardData->invalidBlob = buildInvalidMotionVectorBlob(mvConfig, false);
    forwardData->backward = false;
    VSFilterDependency forwardDeps[]{ { forwardData->node, rpGeneral } };
    forwardNode = vsapi->createVideoFilter2("RIFEMVForward", &forwardData->vi, rifeMVOutputGetFrame, rifeMVOutputFree,
                                            fmParallel, forwardDeps, 1, forwardData.get(), core);
    if (!forwardNode) {
        vsapi->mapSetError(out, "RIFEMV: failed to create forward output filter");
        vsapi->freeNode(forwardData->node);
        vsapi->freeNode(backwardNode);
        vsapi->freeNode(pairNode);
        return;
    }
    forwardData.release();

    vsapi->freeNode(pairNode);
    vsapi->mapConsumeNode(out, "clip", backwardNode, maAppend);
    vsapi->mapConsumeNode(out, "clip", forwardNode, maAppend);
}

static void rifeMVApproxCreateImpl(const VSMap* in, VSMap* out, VSCore* core, const VSAPI* vsapi,
                                   const int maxDelta, const char* functionName) {
    auto pairData{ std::make_unique<RIFEMVApproxPairData>() };
    VSNode* mvClip{};
    VSNode* sourceNode{};
    VSNode* pairNode{};
    VSVideoInfo mvClipVi{};
    bool hasMVClip{};
    bool hasGPUInstance{};
    std::vector<MotionVectorConfig> outputConfigs(maxDelta + 1);
    std::vector<VSNode*> outputNodes;

    try {
        pairData->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        pairData->vi = *vsapi->getVideoInfo(pairData->node);
        int err;

        if (!vsh::isConstantVideoFormat(&pairData->vi) ||
            pairData->vi.format.colorFamily != cfRGB ||
            pairData->vi.format.sampleType != stFloat ||
            pairData->vi.format.bitsPerSample != 32)
            throw "only constant RGB format 32 bit float input supported";

        if (ncnn::create_gpu_instance())
            throw "failed to create GPU instance";
        ++numGPUInstances;
        hasGPUInstance = true;

        auto model_path{ vsapi->mapGetData(in, "model_path", 0, &err) };
        std::string modelPath{ err ? "" : model_path };

        auto gpuId{ vsapi->mapGetIntSaturated(in, "gpu_id", 0, &err) };
        if (err)
            gpuId = ncnn::get_default_gpu_index();

        auto gpuThread{ vsapi->mapGetIntSaturated(in, "gpu_thread", 0, &err) };
        if (err)
            gpuThread = 2;

        auto flowScale{ static_cast<float>(vsapi->mapGetFloat(in, "flow_scale", 0, &err)) };
        if (err)
            flowScale = 1.f;
        FlowResizeMode flowResizeMode{ FlowResizeMode::Auto };
        const auto cpuFlowResize{ vsapi->mapGetIntSaturated(in, "cpu_flow_resize", 0, &err) };
        if (!err)
            flowResizeMode = cpuFlowResize ? FlowResizeMode::ForceCPU : FlowResizeMode::ForceGPU;
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
        const auto mvUseChroma = !!vsapi->mapGetInt(in, "mv_chroma", 0, &err);

        mvClip = vsapi->mapGetNode(in, "mv_clip", 0, &err);
        if (!err) {
            mvClipVi = *vsapi->getVideoInfo(mvClip);
            hasMVClip = true;
        }

        if (hasMVClip) {
            if (!vsh::isConstantVideoFormat(&mvClipVi))
                throw "mv_clip must have a constant format";

            if (mvClipVi.width != pairData->vi.width || mvClipVi.height != pairData->vi.height)
                throw "mv_clip dimensions must match clip";

            if (!mvBitsSpecified)
                mvBits = mvClipVi.format.bitsPerSample;
        }

        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count())
            throw "invalid GPU device";

        if (auto queueCount{ ncnn::get_gpu_info(gpuId).compute_queue_count() }; static_cast<uint32_t>(gpuThread) > queueCount)
            std::cerr << "Warning: gpu_thread is recommended to be between 1 and " << queueCount << " (inclusive)" << std::endl;

        if (gpuThread < 1)
            throw "gpu_thread must be greater than 0";

        if (!std::isfinite(flowScale) || flowScale <= 0.f)
            throw "flow_scale must be greater than 0";

        const auto resolvedModel = resolveRIFEModel(modelPath);
        if (isEarlyUnsupportedRIFEV4Model(resolvedModel.modelPath))
            throw RIFEMVUnsupportedEarlyV4Error;

        if (!supportsMotionVectorExport(resolvedModel))
            throw RIFEMVModelRequirementError;

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

        pairData->mvConfig = createMotionVectorConfig(pairData->vi, hasMVClip ? &mvClipVi : nullptr,
                                                      mvUseChroma, mvBlockSize, mvOverlap, mvPel, 1,
                                                      mvBits, mvHPadding, mvVPadding, mvBlockReduce);
        for (auto delta = 1; delta <= maxDelta; delta++) {
            outputConfigs[delta] = createMotionVectorConfig(pairData->vi, hasMVClip ? &mvClipVi : nullptr,
                                                            mvUseChroma, mvBlockSize, mvOverlap, mvPel, delta,
                                                            mvBits, mvHPadding, mvVPadding, mvBlockReduce);
        }

        if (!vsapi->getVideoFormatByID(&pairData->vi.format, pfGray8, core))
            throw "failed to create output format";

        if (mvClip) {
            vsapi->freeNode(mvClip);
            mvClip = nullptr;
        }

        sourceNode = vsapi->addNodeRef(pairData->node);
        pairData->semaphore = std::make_unique<std::counting_semaphore<>>(gpuThread);
        pairData->rife = std::make_unique<RIFE>(gpuId, flowScale, 1, resolvedModel.rifeV2, resolvedModel.rifeV4, resolvedModel.padding, flowResizeMode);
        loadRIFEModel(*pairData->rife, resolvedModel.modelPath);
    } catch (const char* error) {
        vsapi->mapSetError(out, (std::string(functionName) + ": " + error).c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(mvClip);
        vsapi->freeNode(sourceNode);

        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }

    const auto outputVi = pairData->vi;
    VSFilterDependency pairDeps[]{ { pairData->node, rpGeneral } };
    pairNode = vsapi->createVideoFilter2("RIFEMVApproxPair", &pairData->vi, rifeMVApproxPairGetFrame,
                                         rifeMVApproxPairFree, fmParallel, pairDeps, 1, pairData.get(), core);
    if (!pairNode) {
        vsapi->mapSetError(out, (std::string(functionName) + ": failed to create internal pair filter").c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(sourceNode);
        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }
    pairData.release();

    const auto createOutputNode = [&](const MotionVectorConfig& mvConfig, const bool backward) {
        auto outputData{ std::make_unique<RIFEMVApproxOutputData>() };
        outputData->node = vsapi->addNodeRef(pairNode);
        outputData->sourceNode = vsapi->addNodeRef(sourceNode);
        outputData->vi = outputVi;
        outputData->mvConfig = mvConfig;
        outputData->analysisData = backward ? mvConfig.backwardAnalysisData : mvConfig.forwardAnalysisData;
        outputData->invalidBlob = buildInvalidMotionVectorBlob(mvConfig, backward);
        outputData->backward = backward;
        VSFilterDependency deps[]{ { outputData->node, rpGeneral }, { outputData->sourceNode, rpGeneral } };
        auto node = vsapi->createVideoFilter2(backward ? "RIFEMVApproxBackward" : "RIFEMVApproxForward",
                                              &outputData->vi, rifeMVApproxOutputGetFrame, rifeMVApproxOutputFree,
                                              fmParallel, deps, 2, outputData.get(), core);
        if (!node) {
            vsapi->freeNode(outputData->node);
            vsapi->freeNode(outputData->sourceNode);
            return static_cast<VSNode*>(nullptr);
        }

        outputData.release();
        return node;
    };

    for (auto delta = 1; delta <= maxDelta; delta++) {
        auto backwardNode = createOutputNode(outputConfigs[delta], true);
        if (!backwardNode) {
            vsapi->mapSetError(out, (std::string(functionName) + ": failed to create backward output filter").c_str());
            for (auto* node : outputNodes)
                vsapi->freeNode(node);
            vsapi->freeNode(pairNode);
            vsapi->freeNode(sourceNode);
            return;
        }
        outputNodes.push_back(backwardNode);

        auto forwardNode = createOutputNode(outputConfigs[delta], false);
        if (!forwardNode) {
            vsapi->mapSetError(out, (std::string(functionName) + ": failed to create forward output filter").c_str());
            for (auto* node : outputNodes)
                vsapi->freeNode(node);
            vsapi->freeNode(pairNode);
            vsapi->freeNode(sourceNode);
            return;
        }
        outputNodes.push_back(forwardNode);
    }

    vsapi->freeNode(pairNode);
    vsapi->freeNode(sourceNode);
    for (auto* node : outputNodes)
        vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

static void VS_CC rifeMVApprox2Create(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData,
                                      VSCore* core, const VSAPI* vsapi) {
    rifeMVApproxCreateImpl(in, out, core, vsapi, 2, "RIFEMVApprox2");
}

static void VS_CC rifeMVApprox3Create(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData,
                                      VSCore* core, const VSAPI* vsapi) {
    rifeMVApproxCreateImpl(in, out, core, vsapi, 3, "RIFEMVApprox3");
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.rife", "rife", "Real-Time Intermediate Flow Estimation for Video Frame Interpolation",
                         VS_MAKE_VERSION(9, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    vspapi->registerFunction("RIFE",
                             "clip:vnode;"
                             "factor_num:int:opt;"
                             "factor_den:int:opt;"
                             "fps_num:int:opt;"
                             "fps_den:int:opt;"
                             "model_path:data;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "flow_scale:float:opt;"
                             "cpu_flow_resize:int:opt;"
                             "mv:int:opt;"
                             "mv_backward:int:opt;"
                             "mv_block_size:int:opt;"
                             "mv_overlap:int:opt;"
                             "mv_pel:int:opt;"
                             "mv_delta:int:opt;"
                             "mv_bits:int:opt;"
                             "mv_clip:vnode:opt;"
                             "mv_hpad:int:opt;"
                             "mv_vpad:int:opt;"
                             "mv_block_reduce:int:opt;"
                             "mv_chroma:int:opt;"
                             "sc:int:opt;"
                             "skip:int:opt;"
                             "skip_threshold:float:opt;",
                             "clip:vnode;",
                             rifeCreate, nullptr, plugin);

    vspapi->registerFunction("RIFEMV",
                             "clip:vnode;"
                             "model_path:data;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "flow_scale:float:opt;"
                             "cpu_flow_resize:int:opt;"
                             "mv_block_size:int:opt;"
                             "mv_overlap:int:opt;"
                             "mv_pel:int:opt;"
                             "mv_delta:int:opt;"
                             "mv_bits:int:opt;"
                             "mv_clip:vnode:opt;"
                             "mv_hpad:int:opt;"
                             "mv_vpad:int:opt;"
                             "mv_block_reduce:int:opt;"
                             "mv_chroma:int:opt;",
                             "clip:vnode[];",
                             rifeMVCreate, nullptr, plugin);

    vspapi->registerFunction("RIFEMVApprox2",
                             "clip:vnode;"
                             "model_path:data;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "flow_scale:float:opt;"
                             "cpu_flow_resize:int:opt;"
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
                             rifeMVApprox2Create, nullptr, plugin);

    vspapi->registerFunction("RIFEMVApprox3",
                             "clip:vnode;"
                             "model_path:data;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "flow_scale:float:opt;"
                             "cpu_flow_resize:int:opt;"
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
                             rifeMVApprox3Create, nullptr, plugin);
}

# Synthetic SAD computation

`SAD` means sum of absolute differences. This plugin does not get a matching cost from RIFE itself, so it synthesizes the MVTools `sad` field after a motion vector has already been chosen from RIFE flow or from a composed displacement field.

## Scope

The same `sad` formula is used in all vector-export paths:

- `RIFE(..., mv=1)`: single-output vector export mode, producing either backward or forward vectors depending on `mv_backward`
- `buildMVToolsVectorBlob()`: helper that converts one frame pair plus motion data into the binary MVTools vector blob, including `x`, `y`, and `sad` for each block
- `RIFEMV()`: two-output API that returns both backward and forward vector clips
- `buildMotionVectorBlobFromConfig()`: thin wrapper that builds the same blob using a `MotionVectorConfig` settings object
- `RIFEMVApprox2()` / `RIFEMVApprox3()`: approximate exporters for larger temporal distances, built by composing 2 or 3 adjacent motions
- `buildMotionVectorBlobFromDisplacement()`: helper that builds the blob from already-composed pixel displacements instead of directly from raw flow

Only the way `pixelDx` and `pixelDy` are obtained differs.

## Name glossary

- `round(x)`: C++ `std::lround`, meaning nearest integer with halfway cases rounded away from zero
- `clamp(x, lo, hi)`: limit `x` to the closed interval `[lo, hi]`
- `clampPixel(...)`: frame-edge clamp used for pixel coordinates, so any out-of-range access reads the nearest border pixel
- `bilinearSample(...)`: bilinear interpolation on one float plane after clamping the sample position to the frame bounds
- `luma(r, g, b)`: Rec.709 luma computed as `0.2126 * r + 0.7152 * g + 0.0722 * b`
- `flowX`, `flowY`: horizontal and vertical motion read from the RIFE flow tensor
- `dispX`, `dispY`: horizontal and vertical motion expressed in pixel units rather than raw flow units
- `vx`, `vy`: final stored MVTools vector components, in units of `1 / pel` pixels
- `pixelDx`, `pixelDy`: whole-pixel offsets derived from `vx` and `vy`, used only when computing synthetic SAD

## Inputs

- `current`, `reference`: planar RGB `float32` frames being compared; `current` is the frame whose block is being scored, `reference` is the frame sampled at the motion-shifted position
- `width`, `height`: dimensions of a single plane in pixels
- `stride`: distance between the start of one row and the next, measured in float samples rather than bytes
- `blockSize = mv_block_size`: square block size used for each exported motion vector
- `overlap = mv_overlap`: shared region between neighboring blocks
- `step = blockSize - overlap`: distance between neighboring block origins
- `pel = mv_pel`: subpixel scale used by the stored vector representation; for example, `pel=2` means vector units are half-pixels
- `bits = mv_bits`: synthetic bit depth used to quantize comparison samples, scale exported SAD, and set the MVTools `bitsPerSample` metadata used by downstream threshold scaling; default `8` keeps thresholds and SAD on an 8-bit-equivalent scale unless you override it explicitly
- `sadMultiplier = sad_multiplier`: positive post-scale applied to the final exported SAD values
- `useChroma = mv_chroma`: if `1`, SAD uses all RGB channels; if `0`, SAD uses luma only
- `hPadding = mv_hpad`, `vPadding = mv_vpad`: virtual horizontal and vertical analysis padding used for block placement and vector clamping
- `blockReduce`: how per-pixel motion inside a block is reduced to a single block vector, where `0 = center sample` and `1 = average of the whole block`

Block coordinates are derived from block-grid coordinates `(bx, by)`, where `bx` and `by` are integer block indices in the horizontal and vertical block grid:

```text
blockX = bx * step - hPadding
blockY = by * step - vPadding
```

Any pixel read outside the frame is edge-clamped:

```text
clampPixel(v, limit) = min(max(v, 0), limit - 1)
```

## Motion vector to pixel displacement

### Direct flow path

For normal export, a block vector is obtained from a full-resolution flow plane. Here, a flow plane is a per-pixel motion field at frame resolution. The block reduction is:

- `center`: sample the flow at the block center `(blockX + blockSize / 2, blockY + blockSize / 2)`, edge-clamped
- `average`: average all `blockSize * blockSize` flow samples covered by the block, with each sample position edge-clamped

Channel selection in the 4-channel exported flow tensor is:

- backward vectors use flow channels `0, 1`
- forward vectors use flow channels `2, 3`

The exported vector components are:

```text
vx = round(-2 * flowX * pel)
vy = round(-2 * flowY * pel)
```

They are then clamped so the referenced block stays within the padded analysis area. In other words, the chosen vector is limited so that the motion-shifted block cannot move beyond the configured padded bounds:

```text
minDx = (-hPadding - blockX) * pel
maxDx = (width  - blockSize + hPadding - blockX) * pel
minDy = (-vPadding - blockY) * pel
maxDy = (height - blockSize + vPadding - blockY) * pel

vx = clamp(vx, minDx, maxDx)
vy = clamp(vy, minDy, maxDy)
```

The synthetic SAD uses whole-pixel offsets derived from the clamped vector. This means the stored vector may keep subpixel precision through `pel`, but the actual synthetic SAD lookup always compares integer pixel positions:

```text
pixelDx = round(vx / pel)
pixelDy = round(vy / pel)
```

### Approximate displacement path

For `RIFEMVApprox2/3`, each adjacent-pair flow field is first converted to pixel displacement, meaning motion measured directly in pixels:

```text
dispX = -2 * flowX
dispY = -2 * flowY
```

Multiple displacement fields are composed in sequence. Starting from the first field:

```text
composedX = dispX[0]
composedY = dispY[0]

for each later field i:
    sampleX = x + composedX[x, y]
    sampleY = y + composedY[x, y]
    composedX[x, y] += bilinearSample(dispX[i], sampleX, sampleY)
    composedY[x, y] += bilinearSample(dispY[i], sampleX, sampleY)
```

The bilinear sampler clamps sample coordinates to the frame before interpolation. Here `sampleX` and `sampleY` are floating-point lookup positions reached by following the already-composed motion.

Block reduction is then applied directly to `composedX` and `composedY`, producing pixel-space motion. `pixelBlockDx` and `pixelBlockDy` mean the reduced horizontal and vertical block displacements measured in pixels. Exported vectors are:

```text
vx = round(pixelBlockDx * pel)
vy = round(pixelBlockDy * pel)
```

The same vector clamp is applied as above, and the whole-pixel offsets used by the SAD are again:

```text
pixelDx = round(vx / pel)
pixelDy = round(vy / pel)
```

## Synthetic SAD formula

For each pixel in the block:

```text
currentY   = clampPixel(blockY + y, height)
referenceY = clampPixel(currentY + pixelDy, height)
currentX   = clampPixel(blockX + x, width)
referenceX = clampPixel(currentX + pixelDx, width)

currentIndex   = currentY   * stride + currentX
referenceIndex = referenceY * stride + referenceX
scale = (1 << bits) - 1
q(v) = round(v * scale) / scale
```

Here `currentX/currentY` are the edge-clamped coordinates inside the source block, `referenceX/referenceY` are the corresponding motion-shifted coordinates in the reference frame, and `currentIndex/referenceIndex` are row-major array indices into the planar float buffers. `scale` is the maximum integer sample value for the chosen synthetic SAD bit depth. `q(v)` is the synthetic-bit-depth quantizer applied to comparison samples before the absolute difference is measured.

If `mv_chroma=1`, the per-pixel contribution is:

```text
round((abs(q(Rc) - q(Rr)) + abs(q(Gc) - q(Gr)) + abs(q(Bc) - q(Br))) * scale)
```

If `mv_chroma=0`, RGB is converted to luma first:

```text
luma(r, g, b) = 0.2126 * r + 0.7152 * g + 0.0722 * b
```

and the per-pixel contribution is:

```text
round(abs(luma(q(Rc), q(Gc), q(Bc)) - luma(q(Rr), q(Gr), q(Br))) * scale)
```

The block `sad` is the sum of those rounded per-pixel contributions over the full `blockSize x blockSize` block. This is a synthetic matching cost intended to populate MVTools metadata, not a native RIFE confidence or loss value.

After that block sum is computed, the plugin applies the exported SAD calibration multiplier:

```text
sad = round(sad * sadMultiplier)
```

With the default `sad_multiplier=1.0`, the multiplier itself does not add any extra calibration beyond the chosen SAD bit scale.

With the default `bits=8`, exported SAD stays on an 8-bit-equivalent scale even when the metadata source clip is 10-bit or higher. Because the comparison samples are also quantized to that same synthetic bit depth before differencing and the exported MVTools bit-depth metadata matches that same scale, downstream `thsad` and `thscd1` thresholds stay aligned with the exported SAD values.

Important implementation detail: rounding happens per pixel before accumulation, not once at the end.

## Block-size normalization and frame props

The exported `VECTOR.sad` values stored in `MVTools_vectors` remain raw block SAD values. They therefore grow with larger block sizes, which is expected and required for MVTools compatibility because MVTools already rescales user thresholds such as `thsad` and `thscd1` using block size, chroma mode, and `bitsPerSample`.

That means:

- `RIFEMV_AvgSad` is the raw mean exported block SAD and will generally increase with larger blocks.
- `RIFEMV_AvgSad8x8` is the same average converted back into the 8x8-equivalent threshold space used by MVTools user parameters. This is the property to compare across different block sizes when you want a more stable threshold-oriented metric.

Because MVTools performs its own block-size normalization internally, changing the actual exported `VECTOR.sad` values to an 8x8-equivalent scale would break downstream threshold behavior.

## Invalid vectors

If no valid reference frame exists, the vector is marked invalid and uses:

```text
vx = 0
vy = 0
sad = round(blockSize * blockSize * (1 << bits) * sadMultiplier)
```

This sentinel is stored directly without running the per-pixel SAD loop.
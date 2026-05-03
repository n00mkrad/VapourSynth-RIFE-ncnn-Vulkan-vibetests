# About this modification

This modified build of the VapourSynth RIFE plugin adds MVTools-compatible motion-vector export.

Instead of only generating interpolated frames, the plugin can now export motion vectors in the same binary frame-property format used by MVTools:

- `MVTools_MVAnalysisData`
- `MVTools_vectors`

That makes it possible to feed RIFE-derived motion into MVTools consumers such as `mv.Mask`, `mv.Flow`, `mv.FlowBlur`, and, with suitable settings, degrain-style functions.

## What this build adds

- `rife.RIFE(..., mv=1)`
  Exports a single MVTools vector clip, either backward or forward depending on `backward`.

- `rife.RIFEMV(...)`
  Returns both backward and forward vector clips at once.
  The current implementation shares one inference pass per adjacent frame pair.

- `rife.RIFEMVApprox2(...)`
  Returns approximate vectors for deltas 1 and 2 by composing adjacent motions.

- `rife.RIFEMVApprox3(...)`
  Returns approximate vectors for deltas 1, 2, and 3 by composing adjacent motions.

## Important limitations

- Motion-vector export supports `rife-v3.1`, `rife-v3.9`, and `rife-v4.2+` model families.
- Legacy `rife-v4` as well as `rife-v4.0` and `rife-v4.1` are not supported for motion-vector export.
- Motion-vector APIs accept either constant-format `RGBS` or constant-format `YUV`. Non-`RGBS` `YUV` input is converted internally to `RGBS` for RIFE inference.
- MVTools usually operates on a different clip, often `YUV420P8`. `meta_clip` is still optional and can be used explicitly as the metadata source.
- If the input is non-`RGBS` and `meta_clip` is omitted, the plugin now uses the original input clip as the metadata source automatically.
- Vector clips are dummy `Gray8` clips whose pixel values are not meaningful. The motion data lives in frame properties.
- Exported motion-vector frames also include `RIFEMV_AvgSad` as an integer frame property containing the raw average exported block SAD, `RIFEMV_AvgSad8x8` as an integer 8x8-equivalent average SAD in MVTools threshold space, plus `RIFEMV_AvgAbsDx`, `RIFEMV_AvgAbsDy`, and `RIFEMV_AvgAbsMotion` as float frame properties containing the average absolute horizontal motion, average absolute vertical motion, and average absolute motion magnitude for that frame.
- Do not resize or colorspace-convert the exported vector clips after creation.

## API changes in `rife.RIFE`

`rife.RIFE` still performs normal interpolation, but it now also supports single-direction motion-vector export.

### Signature

```python
core.rife.RIFE(clip, factor_num=2, factor_den=1, fps_num=None, fps_den=None, model_path=..., gpu_id=default_gpu, gpu_thread=2, shared_flow_inflight=None, flow_scale=1.0, cpu_flow_resize=None, mv=0, backward=1, blksize_x=16, blksize_y=None, overlap_x=None, overlap_y=None, pel=1, delta=1, bits=8, sad_multiplier=1.0, meta_clip=None, matrix_in_s="709", range_in_s="full", hpad=0, vpad=0, block_reduce=1, chroma=0, blksize_int_x=None, blksize_int_y=None, sc=0, skip=0, skip_threshold=60.0)
```

### New or changed arguments

- `flow_scale`
  Scales the image before flow estimation and rescales vectors back to the original image coordinates. Smaller values can reduce cost and can sometimes behave better on large motion.
  `flow_scale` replaces the `uhd` bool parameter used in the original plugin. To match the old behavior, use `0.5` for uhd=True or `1.0` (default) for uhd=False.
  Accepted values are restricted to: `0.25`, `0.5`, `1.0`, `2.0`, `4.0`.

- `cpu_flow_resize`
  Debug control for the flow upsampling path used by motion-vector export.
  Omit this argument for automatic behavior (GPU resize when available, CPU fallback on failure).
  `False` (`0`) forces the GPU resize path only.
  `True` (`1`) forces the CPU resize fallback path.

- `shared_flow_inflight`
  Global in-flight cap for motion-vector flow inference shared across filter instances on the same GPU.
  This only affects motion-vector paths (`mv=1`, `RIFEMV`, `RIFEMVApprox2`, `RIFEMVApprox3`).
  Default: GPU compute queue count.
  Lower values can reduce CPU contention; higher values can increase throughput on some setups.
  When explicitly set, local admission is relaxed to at least this value (`max(gpu_thread, shared_flow_inflight)`) so the shared cap remains the primary limiter.

- `mv`
  Enables MVTools vector export mode when set to `1`. In this mode, the output is a vector clip, not an interpolated image clip.

- `backward`
  Selects vector direction for `mv=1`.
  `1` = backward vectors.
  `0` = forward vectors.

- `blksize_x`, `blksize_y`
  Exported MVTools block size on each axis.
  Default: `blksize_x=16`, `blksize_y=blksize_x`.

- `overlap_x`, `overlap_y`
  Exported MVTools overlap on each axis.
  Default: `overlap_x=blksize_x // 2`, `overlap_y=blksize_y // 2`.

- `pel`
  MVTools pel value written to metadata and used for vector scaling.
  Default: `1`.

- `delta`
  Temporal distance written to `nDeltaFrame` in the MV metadata for single-output export.
  Default: `1`.

- `bits`
  Synthetic bit depth used when computing exported SAD values.
  Default: `8`.
  Leaving this at `8` keeps exported SAD on an 8-bit scale regardless of source or `meta_clip` bit depth.
  Set a higher value only if you explicitly want larger SAD values that track a higher-bit-depth scale.
  This also sets the exported MVTools `bitsPerSample` metadata so downstream filters scale `thsad` and `thscd1` against the same SAD range.

- `sad_multiplier`
  Positive multiplier applied to the final synthetic SAD values written into exported MVTools vectors.
  Default: `1.0`.
  This scales valid block SADs, invalid-frame sentinel SADs, and therefore `RIFEMV_AvgSad`.
  It does not affect motion estimation or exported vector `x`/`y` components.

- `meta_clip`
  Metadata-source clip for MVTools compatibility.
  This should usually be the actual clip you will feed to MVTools, for example the original `YUV420P8` source.
  If omitted and `clip` is non-`RGBS`, the plugin uses the original input clip automatically.
  This parameter is named `meta_clip` instead of plain `clip` to avoid conflicting with the primary input clip parameter.

- `matrix_in_s`
  Input matrix used when the MV API receives a non-`RGBS` `YUV` clip and performs internal conversion to `RGBS`.
  Default: `"709"`.

- `range_in_s`
  Input range used for that same internal `YUV` -> `RGBS` conversion.
  Default: `"full"`.

- `hpad`, `vpad`
  Horizontal and vertical padding written into MV metadata and used for vector clamping.
  These should match the corresponding `mv.Super` settings when relevant.

- `block_reduce`
  Controls how dense RIFE flow is reduced to one block vector.
  `0` = center sample.
  `1` = average over the whole block.
  Default: `1`.

- `chroma`
  If enabled, synthetic SAD includes all RGB channels. Otherwise it uses luma only.

- `blksize_int_x`, `blksize_int_y`
  MV-export-only internal block size used for inference reduction.
  Defaults: `blksize_int_x=blksize_x`, `blksize_int_y=blksize_int_x`.
  The plugin derives independent horizontal and vertical resize ratios from `blksize_int_x / blksize_x` and `blksize_int_y / blksize_y`, resizes the RGBS inference clip by those ratios, scales `overlap_x`, `overlap_y`, `hpad`, and `vpad` on their respective axes, and rejects the request if any derived internal width, height, overlap, or padding value would be non-integer.
  Exported MVTools metadata and the dummy `Gray8` vector clip stay at the original clip resolution, and the exported vectors are scaled back up to that original coordinate space.
  `blksize_int_x` and `blksize_int_y` are only accepted by `rife.RIFE` when `mv=1`.

### Typical single-direction export

```python
mvbw = core.rife.RIFE(clip, model_path=rife_mdl, mv=1, backward=1, matrix_in_s="709", range_in_s="full")
```

Use this mode when a function expects only one vector clip.

## `rife.RIFEMV`

`rife.RIFEMV` is the convenience function for the common delta-1 case.

### Signature

```python
mvbw, mvfw = core.rife.RIFEMV(clip, model_path=..., gpu_id=default_gpu, gpu_thread=2, shared_flow_inflight=None, flow_scale=1.0, cpu_flow_resize=None, perf_stats=False, blksize_x=16, blksize_y=None, overlap_x=None, overlap_y=None, pel=1, delta=1, bits=8, sad_multiplier=1.0, meta_clip=None, matrix_in_s="709", range_in_s="full", hpad=0, vpad=0, block_reduce=1, chroma=0, blksize_int_x=None, blksize_int_y=None)
```

### Return value

`RIFEMV` returns `clip:vnode[]` with this ordering:

```python
mvbw, mvfw = core.rife.RIFEMV(...)
```

- first output: backward vectors
- second output: forward vectors

`cpu_flow_resize` has the same meaning as in `rife.RIFE`:
- omitted = automatic (GPU resize with CPU fallback)
- `0`/`False` = force GPU resize
- `1`/`True` = force CPU resize

`perf_stats` enables per-filter performance timing. When enabled, a summary is printed to `stderr` when the filter instance is freed (end of clip processing).

`perf_stats` now reports:
- `semaphore_wait_ms` total wait
- `local_wait_ms` wait on per-filter `gpu_thread` limiter
- `shared_wait_ms` wait on the cross-instance `shared_flow_inflight` limiter

`sad_multiplier` has the same meaning as in `rife.RIFE(..., mv=1)`:
- positive float, default `1.0`
- scales exported synthetic SAD values only
- does not affect vector estimation or `RIFEMV_AvgAbs*` properties

### Recommended usage

```python
mvbw, mvfw = core.rife.RIFEMV(clip, model_path=rife_mdl, matrix_in_s="709", range_in_s="full")

mask = core.mv.Mask(clip, mvfw, kind=5, ml=100.0)
```

### Example with Degrain1

```python
sup = core.mv.Super(clip, pel=1, hpad=0, vpad=0, levels=1)
mvbw, mvfw = core.rife.RIFEMV(clip, model_path=rife_mdl, matrix_in_s="709", range_in_s="full", pel=1, hpad=0, vpad=0)

den = core.mv.Degrain1(clip, sup, mvbw, mvfw, thsad=500)
```

When using MVTools consumers that depend on `pel`, `hpad`, or `vpad`, keep those values aligned between `mv.Super(...)` and the RIFE exporter.

## `rife.RIFEMVApprox2` and `rife.RIFEMVApprox3`

These functions generate approximate larger-delta motion by composing adjacent frame-to-frame displacements.

They are useful when you want delta-2 or delta-3 vectors without running separate direct exporters for each temporal distance.

### `RIFEMVApprox2`

```python
outputs = core.rife.RIFEMVApprox2(clip, model_path=rife_mdl, matrix_in_s="709", range_in_s="full")
```

Output order:

```python
bw1, fw1, bw2, fw2 = core.rife.RIFEMVApprox2(...)
```

- `bw1`, `fw1`: approximate delta-1 vectors
- `bw2`, `fw2`: approximate delta-2 vectors

### `RIFEMVApprox3`

```python
outputs = core.rife.RIFEMVApprox3(clip, model_path=rife_mdl, matrix_in_s="709", range_in_s="full")
```

Output order:

```python
bw1, fw1, bw2, fw2, bw3, fw3 = core.rife.RIFEMVApprox3(...)
```

- `bw1`, `fw1`: approximate delta-1 vectors
- `bw2`, `fw2`: approximate delta-2 vectors
- `bw3`, `fw3`: approximate delta-3 vectors

### Shared arguments

`RIFEMVApprox2` and `RIFEMVApprox3` accept the same arguments as `RIFEMV`, except they do not expose `delta` because each function has a fixed maximum delta built into it.

`blksize_int_x` and `blksize_int_y` have the same meaning in `RIFEMV`, `RIFEMVApprox2`, and `RIFEMVApprox3` as they do for `rife.RIFE(..., mv=1)`.

### Example with Degrain2

```python
sup = core.mv.Super(clip, pel=1, hpad=0, vpad=0, levels=1)
bw1, fw1, bw2, fw2 = core.rife.RIFEMVApprox2(clip, model_path=rife_mdl, matrix_in_s="709", range_in_s="full")

den = core.mv.Degrain2(clip, sup, bw1, fw1, bw2, fw2, thsad=500)
```

## Practical notes

- Use `block_reduce=1` as the default starting point for degraining.
- You can pass YUV clips directly; internal conversion to RGBS is done automatically for MV inference.
- `meta_clip` is optional. For non-`RGBS` input it is auto-inferred from the original input clip when omitted; pass `meta_clip` explicitly only if you want a different metadata source.
- Keep `pel`, `hpad`, and `vpad` consistent with the `mv.Super` clip you use downstream.
- If a function only needs one direction, `rife.RIFE(..., mv=1)` is enough.
- If you need both directions for delta 1, prefer `rife.RIFEMV(...)`.
- If you need approximate delta 2 or 3 vectors, use `rife.RIFEMVApprox2(...)` or `rife.RIFEMVApprox3(...)`.

## Summary

This modification turns RIFE into both an interpolator and a motion-vector source for MVTools workflows.

The key idea is:

- RIFE still runs on `RGBS` internally (with optional automatic YUV -> RGBS conversion in MV APIs)
- MVTools still consumes its own vector-clip format
- this build bridges the two by exporting MVTools-compatible binary vector properties

That makes it possible to reuse RIFE optical flow in more traditional VapourSynth motion-processing pipelines.

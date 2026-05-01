# About this modification

This modified build of the VapourSynth RIFE plugin adds MVTools-compatible motion-vector export.

Instead of only generating interpolated frames, the plugin can now export motion vectors in the same binary frame-property format used by MVTools:

- `MVTools_MVAnalysisData`
- `MVTools_vectors`

That makes it possible to feed RIFE-derived motion into MVTools consumers such as `mv.Mask`, `mv.Flow`, `mv.FlowBlur`, and, with suitable settings, degrain-style functions.

## What this build adds

- `rife.RIFE(..., mv=1)`
  Exports a single MVTools vector clip, either backward or forward depending on `mv_backward`.

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
- The analysis input to RIFE must still be constant-format `RGBS`.
- MVTools usually operates on a different clip, often `YUV420P8`. When that is the case, pass `mv_clip=clip` so the exported MV metadata uses the correct bit depth and chroma subsampling.
- Vector clips are dummy `Gray8` clips whose pixel values are not meaningful. The motion data lives in frame properties.
- Do not resize or colorspace-convert the exported vector clips after creation.

## API changes in `rife.RIFE`

`rife.RIFE` still performs normal interpolation, but it now also supports single-direction motion-vector export.

### Signature

```python
core.rife.RIFE(clip, factor_num=2, factor_den=1, fps_num=None, fps_den=None, model_path=..., gpu_id=default_gpu, gpu_thread=2, flow_scale=1.0, cpu_flow_resize=None, mv=0, mv_backward=1, mv_block_size=16, mv_overlap=8, mv_pel=1, mv_delta=1, mv_bits=8, mv_clip=None, mv_hpad=0, mv_vpad=0, mv_block_reduce=1, mv_chroma=0, sc=0, skip=0, skip_threshold=60.0)
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

- `mv`
  Enables MVTools vector export mode when set to `1`. In this mode, the output is a vector clip, not an interpolated image clip.

- `mv_backward`
  Selects vector direction for `mv=1`.
  `1` = backward vectors.
  `0` = forward vectors.

- `mv_block_size`
  Block size used for exported MVTools vectors.
  Default: `16`.

- `mv_overlap`
  Block overlap used for exported MVTools vectors.
  Default: `8`.

- `mv_pel`
  MVTools pel value written to metadata and used for vector scaling.
  Default: `1`.

- `mv_delta`
  Temporal distance written to `nDeltaFrame` in the MV metadata for single-output export.
  Default: `1`.

- `mv_bits`
  Synthetic bit depth used when computing exported SAD values.
  Default: `8`.
  If `mv_clip` is given and `mv_bits` is omitted, the plugin may derive bit depth from `mv_clip`.

- `mv_clip`
  Metadata-source clip for MVTools compatibility.
  This should usually be the actual clip you will feed to MVTools, for example the original `YUV420P8` source.

- `mv_hpad`, `mv_vpad`
  Horizontal and vertical padding written into MV metadata and used for vector clamping.
  These should match the corresponding `mv.Super` settings when relevant.

- `mv_block_reduce`
  Controls how dense RIFE flow is reduced to one block vector.
  `0` = center sample.
  `1` = average over the whole block.
  Default: `1`.

- `mv_chroma`
  If enabled, synthetic SAD includes all RGB channels. Otherwise it uses luma only.

### Typical single-direction export

```python
rife_in_clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709", range_in_s="full")

mvbw = core.rife.RIFE(rife_in_clip, model_path=rife_mdl, mv=1, mv_backward=1, mv_clip=clip)
```

Use this mode when a function expects only one vector clip.

## `rife.RIFEMV`

`rife.RIFEMV` is the convenience function for the common delta-1 case.

### Signature

```python
mvbw, mvfw = core.rife.RIFEMV(clip, model_path=..., gpu_id=default_gpu, gpu_thread=2, flow_scale=1.0, cpu_flow_resize=None, mv_block_size=16, mv_overlap=8, mv_pel=1, mv_delta=1, mv_bits=8, mv_clip=None, mv_hpad=0, mv_vpad=0, mv_block_reduce=1, mv_chroma=0)
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

### Recommended usage

```python
rife_in_clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709", range_in_s="full")

mvbw, mvfw = core.rife.RIFEMV(rife_in_clip, model_path=rife_mdl, mv_clip=clip)

mask = core.mv.Mask(clip, mvfw, kind=5, ml=100.0)
```

### Example with Degrain1

```python
rife_in_clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709", range_in_s="full")

sup = core.mv.Super(clip, pel=1, hpad=0, vpad=0, levels=1)
mvbw, mvfw = core.rife.RIFEMV(rife_in_clip, model_path=rife_mdl, mv_clip=clip, mv_pel=1, mv_hpad=0, mv_vpad=0)

den = core.mv.Degrain1(clip, sup, mvbw, mvfw, thsad=500)
```

When using MVTools consumers that depend on `pel`, `hpad`, or `vpad`, keep those values aligned between `mv.Super(...)` and the RIFE exporter.

## `rife.RIFEMVApprox2` and `rife.RIFEMVApprox3`

These functions generate approximate larger-delta motion by composing adjacent frame-to-frame displacements.

They are useful when you want delta-2 or delta-3 vectors without running separate direct exporters for each temporal distance.

### `RIFEMVApprox2`

```python
outputs = core.rife.RIFEMVApprox2(rife_in_clip, model_path=rife_mdl, mv_clip=clip)
```

Output order:

```python
bw1, fw1, bw2, fw2 = core.rife.RIFEMVApprox2(...)
```

- `bw1`, `fw1`: approximate delta-1 vectors
- `bw2`, `fw2`: approximate delta-2 vectors

### `RIFEMVApprox3`

```python
outputs = core.rife.RIFEMVApprox3(rife_in_clip, model_path=rife_mdl, mv_clip=clip)
```

Output order:

```python
bw1, fw1, bw2, fw2, bw3, fw3 = core.rife.RIFEMVApprox3(...)
```

- `bw1`, `fw1`: approximate delta-1 vectors
- `bw2`, `fw2`: approximate delta-2 vectors
- `bw3`, `fw3`: approximate delta-3 vectors

### Shared arguments

`RIFEMVApprox2` and `RIFEMVApprox3` accept the same arguments as `RIFEMV`, except they do not expose `mv_delta` because each function has a fixed maximum delta built into it.

### Example with Degrain2

```python
rife_in_clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709", range_in_s="full")

sup = core.mv.Super(clip, pel=1, hpad=0, vpad=0, levels=1)
bw1, fw1, bw2, fw2 = core.rife.RIFEMVApprox2(rife_in_clip, model_path=rife_mdl, mv_clip=clip)

den = core.mv.Degrain2(clip, sup, bw1, fw1, bw2, fw2, thsad=500)
```

## Practical notes

- Use `mv_block_reduce=1` as the default starting point for degraining.
- Use `mv_clip=clip` whenever the RIFE analysis clip is RGBS but the MVTools consumer clip is YUV.
- Keep `mv_pel`, `mv_hpad`, and `mv_vpad` consistent with the `mv.Super` clip you use downstream.
- If a function only needs one direction, `rife.RIFE(..., mv=1)` is enough.
- If you need both directions for delta 1, prefer `rife.RIFEMV(...)`.
- If you need approximate delta 2 or 3 vectors, use `rife.RIFEMVApprox2(...)` or `rife.RIFEMVApprox3(...)`.

## Summary

This modification turns RIFE into both an interpolator and a motion-vector source for MVTools workflows.

The key idea is:

- RIFE still runs on `RGBS`
- MVTools still consumes its own vector-clip format
- this build bridges the two by exporting MVTools-compatible binary vector properties

That makes it possible to reuse RIFE optical flow in more traditional VapourSynth motion-processing pipelines.

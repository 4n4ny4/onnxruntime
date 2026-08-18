# YOLO26n LiteRT.js vs ONNX Runtime WebGPU benchmark

This development page compares matched browser inference for LiteRT.js and ONNX Runtime Web. It is intentionally limited
to the raw YOLO26n detection head—no NMS or end-to-end postprocessing—and does not commit model or input binaries.

## Required artifacts

| Artifact                                    | Required SHA256                                                    | Shape                           |
| ------------------------------------------- | ------------------------------------------------------------------ | ------------------------------- |
| Official `yolo26n.pt` checkpoint provenance | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | N/A                             |
| Launch-era raw ONNX                         | `5b551efe169f8656e883a677740bd92e2eb73bdd38ea0b02d3f596c9a9d3e807` | `[1,3,640,640]` → `[1,84,8400]` |
| Raw FP32 TFLite                             | `aec199b2383caf5fe356afbf966d63364bc1b66e0620b599a177501fd99662cf` | `[1,640,640,3]` → `[1,84,8400]` |
| Deterministic input NPY                     | `a95ffe2ea506d816578be4da69c1fe76b9ef2a658ab79e3fc101c3fdbb5cb60b` | `[1,3,640,640]`, float32        |

The page accepts local files or URLs and validates the selected bytes in-browser. URL assets must be same-origin or return
appropriate CORS headers.

## Build and run

Use Node.js 20 or newer. Install the repository JavaScript tooling and package dependencies using the existing lockfiles:

```powershell
Set-Location js
npm ci
Set-Location common
npm ci
Set-Location ..\web
npm ci
```

Build the repository's native C++ WebGPU EP Asyncify WASM module and the matching web bundle. The benchmark build check
rejects JSEP artifacts so engine identity cannot silently change between runs.

```powershell
Copy-Item ..\..\build_webgpu_asyncify\Release\ort-wasm-simd-threaded.asyncify.* .\dist\
npm run build -- --bundle-mode=perf --webgpu-ep
npm run test:benchmark:yolo26-webgpu
npm run serve:benchmark:yolo26-webgpu
```

Open the printed localhost URL in current Chrome or Edge. `npm run test:benchmark:yolo26-webgpu` builds the TypeScript
entry point and smoke-loads the page, ORT bundle, native WebGPU EP WASM module, LiteRT.js module, and LiteRT.js WASM module over
HTTP.

## Matched methodology

- **Runtimes:** current-repository ONNX Runtime native C++ WebGPU EP/Asyncify WASM bundle and exactly
  `@litertjs/core@2.5.3`. Engine identity is embedded in the result.
- **One device:** the page creates one high-performance `GPUDevice` and passes that exact object to the native ORT
  WebGPU EP and LiteRT.js `Environment({webGpuDevice})`.
- **FP32 primary:** LiteRT.js compiles with `gpuOptions.precision='fp32'`. LiteRT.js echoes the precision request but has
  no post-compilation precision query, so the page records that limitation and relies on the output parity gate rather
  than claiming hidden precision introspection.
- **Full GPU only:** ORT sets `session.disable_cpu_ep_fallback=1`; session creation fails if any graph node is assigned to
  CPU. LiteRT.js must retain `accelerator='webgpu'` and report `isFullyAccelerated === true`. Hybrid results are rejected.
- **Same semantic data:** the selected NCHW input is transposed to NHWC once for TFLite, outside all timed regions.
  Outputs must be `[1,84,8400]` or `[1,8400,84]`; the latter is normalized after readback and outside timing. The
  immutable TFLite export stores its four box channels normalized to image size, so those channels are multiplied by 640
  after readback before parity. That documented semantic normalization is also outside timing.
- **Correctness first:** comparison mode compiles and runs each runtime sequentially, disposes it, normalizes its output,
  and requires max absolute error ≤ `1e-3` and p99 absolute error ≤ `1e-4` before any timed round begins. It reports max,
  p99, and mean absolute error.
- **Synchronized timing:** default input tensors are uploaded once and reused. Every sample covers the inference call
  through explicit CPU output readback (`Tensor.getData()` for ORT and `Tensor.data()` for LiteRT.js), which is a real GPU
  completion point. Enqueue-only timing is never reported.
- **Stage separation:** runtime/model initialization, model compilation, input conversion/upload, first inference,
  warmups, steady-state inference, normalization, and UI rendering are separately handled. Only steady-state inference
  plus output readback enters the sample arrays.
- **Rounds and order:** the qualified default is five fresh model compilations per arm, each with 20 warmups and 100 timed
  samples. Arms run sequentially with queue completion and disposal between them. Qualified runs randomize the first
  runtime at launch and alternate thereafter, and wait at least one second after each fresh compilation before inference;
  deterministic and fixed-order development modes remain available.
- **Statistics:** aggregate and per-round p50, p90, mean, sample standard deviation, IQR, min, max, and mean-derived
  inferences/second are reported. Downloaded JSON retains every raw latency sample.

ORT native WebGPU graph capture is off by default and can be enabled with `?graphCapture=1`. It re-encodes recorded
pipeline, bind-group, and dispatch records while preserving stable external I/O; it does not resubmit a
`GPUCommandBuffer` and is unrelated to native Python graph capture. The result records whether capture was enabled.

## Qualification and limitations

The UI labels results **presentation-qualified** only when all immutable hashes, iteration counts, randomized order,
stabilization, explicit-FP32, same-device, full-delegation, output-shape, and parity gates pass. Individual runtime runs,
reduced counts, per-run host upload, custom artifacts, or missing parity are labeled **development-only**.

WebGPU requires a secure context (HTTPS or localhost). This benchmark loads unthreaded LiteRT.js and forces one ORT WASM
thread, so it does not require cross-origin isolation. Cancellation is cooperative between samples because neither
runtime exposes cancellation for an in-flight WebGPU inference.

### Local NVIDIA naming

The physical adapter is an NVIDIA TITAN V (GV100, Volta), while WebGPU reports `vendor=nvidia` and
`architecture=pascal`. Results must preserve the browser-reported string without claiming Pascal hardware validation.

### Native Intel blocker

The native Intel comparison is blocked, not a result. LiteRT 2.1.6 on Windows selected NVIDIA TITAN V despite its public
preferred-device selector, and direct Intel device injection crashed. ONNX Runtime was proven on Intel UHD 770, but no
native Intel win, tie, or loss is reported.

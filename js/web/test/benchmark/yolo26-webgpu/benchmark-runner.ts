// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import type * as OrtApi from 'onnxruntime-common';

import {
  ArmName,
  calculateStatistics,
  CANONICAL_OUTPUT_SHAPE,
  compareOutputs,
  EXPECTED_INPUT_SHA256,
  EXPECTED_ONNX_SHA256,
  EXPECTED_TFLITE_SHA256,
  getRoundOrder,
  LITERT_INPUT_SHAPE,
  normalizeOutput,
  OFFICIAL_CHECKPOINT_SHA256,
  OrderStrategy,
  ORT_INPUT_SHAPE,
  parseFloat32Npy,
  QUALIFIED_MAX_ABSOLUTE_ERROR,
  QUALIFIED_P99_ABSOLUTE_ERROR,
  requireShape,
  scaleYoloBoxChannels,
  sha256Hex,
  Statistics,
  TRANSPOSED_OUTPUT_SHAPE,
  transposeNchwToNhwc,
  validateSha256,
} from './benchmark-utils';
import { WebGpuApiTrace, WebGpuTraceMarker } from './webgpu-api-trace';

declare const BENCHMARK_BUILD_INFO: {
  readonly gitCommit: string;
  readonly litertVersion: string;
  readonly ortBundle: string;
  readonly ortEngine: 'jsep-webgpu' | 'native-webgpu-ep';
};

declare global {
  interface Window {
    ort?: typeof OrtApi;
  }
}

export const benchmarkBuildInfo = BENCHMARK_BUILD_INFO;

export type BenchmarkMode = 'comparison' | ArmName;
export type InputBoundary = 'gpu-resident' | 'per-run-upload';

export interface AssetSelection {
  readonly file: File | null;
  readonly url: string;
  readonly expectedSha256: string;
}

export interface BenchmarkRequest {
  readonly mode: BenchmarkMode;
  readonly onnx: AssetSelection;
  readonly tflite: AssetSelection;
  readonly input: AssetSelection;
  readonly warmups: number;
  readonly iterations: number;
  readonly rounds: number;
  readonly order: OrderStrategy;
  readonly stabilizationMilliseconds: number;
  readonly inputBoundary: InputBoundary;
  readonly graphCapture: boolean;
  readonly queueSyncDiagnostic: boolean;
  readonly webGpuTrace: boolean;
  readonly wallDecomposition: boolean;
  readonly gpuResidentOutput: boolean;
}

export interface ProgressUpdate {
  readonly completed: number;
  readonly total: number;
  readonly message: string;
}

export interface BenchmarkHooks {
  readonly isCancelled: () => boolean;
  readonly onProgress: (update: ProgressUpdate) => void;
  readonly onStatus: (message: string) => void;
}

interface LoadedAsset {
  readonly bytes: Uint8Array;
  readonly sha256: string;
  readonly source: string;
  readonly sizeBytes: number;
}

interface LoadedAssets {
  readonly onnx?: LoadedAsset;
  readonly tflite?: LoadedAsset;
  readonly input: LoadedAsset;
  readonly inputNchw: Float32Array;
  readonly inputNhwc?: Float32Array;
  readonly inputLayoutConversionMilliseconds?: number;
  readonly inputParsingMilliseconds: number;
}

interface InferenceOutput {
  readonly data: Float32Array;
  readonly shape: readonly number[];
  readonly instrumentation?: RunInstrumentation;
}

interface RunInstrumentation {
  readonly schedulingMilliseconds: number;
  readonly gpuCompletionMilliseconds?: number;
  readonly readbackMilliseconds: number;
  readonly totalMilliseconds: number;
  readonly webGpuApiTrace?: Record<string, unknown>;
  readonly wallDecomposition?: {
    readonly encodeSetupMilliseconds: number;
    readonly encodeSpanMilliseconds: number;
    readonly gpuDrainMilliseconds: number;
    readonly readbackMilliseconds: number;
    readonly copyOutPostMilliseconds: number;
    readonly bucketSumMilliseconds: number;
    readonly wallMilliseconds: number;
    readonly bucketCoverageRatio: number;
    readonly preReadbackSubmitCount: number;
    readonly totalSubmitCount: number;
    readonly computePassCount: number;
    readonly copyBufferToBufferCount: number;
    readonly mapAsyncCount: number;
  };
}

interface RuntimeProof {
  readonly accelerator: string;
  readonly deviceApiProof: string;
  readonly deviceIdentity: string;
  readonly deviceReferenceVerified: boolean;
  readonly deviceSnapshot: Record<string, unknown>;
  readonly fallbackDisabled?: boolean;
  readonly fullyAccelerated?: boolean;
  readonly inputShape: readonly number[];
  readonly outputShape: readonly number[];
  readonly outputLocation: string;
  readonly precision: string;
  readonly webGpuApiTrace?: Record<string, unknown>;
}

interface ArmRunner {
  readonly arm: ArmName;
  readonly compileMilliseconds: number;
  readonly proof: RuntimeProof;
  enableWallDecomposition?(): void;
  runOnce(readbackOutput?: boolean): Promise<InferenceOutput>;
  dispose(): Promise<void>;
}

interface RoundResult {
  readonly arm: ArmName;
  readonly round: number;
  readonly orderInRound: number;
  readonly compileMilliseconds: number;
  readonly firstInferenceMilliseconds: number;
  readonly samplesMilliseconds: readonly number[];
  readonly statistics: Statistics;
  readonly instrumentation?: {
    readonly firstInference: RunInstrumentation;
    readonly timedSamples: readonly RunInstrumentation[];
  };
  readonly wallDecomposition?: {
    readonly samplesMilliseconds: readonly number[];
    readonly statistics: Statistics;
    readonly instrumentation: readonly RunInstrumentation[];
  };
  readonly runtimeProof: RuntimeProof;
}

interface CorrectnessArmResult {
  readonly compileMilliseconds: number;
  readonly inferenceMilliseconds: number;
  readonly sourceShape: readonly number[];
  readonly normalizedShape: readonly number[];
  readonly normalization: string;
  readonly runtimeProof: RuntimeProof;
}

interface SharedGpu {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly diagnostics: Record<string, unknown>;
  readonly origin: 'caller-created-shared';
}

class CancelledError extends Error {
  constructor() {
    super('Benchmark cancelled. The active inference was allowed to finish before cancellation.');
    this.name = 'CancelledError';
  }
}

class ProgressTracker {
  private completed = 0;

  constructor(
    private readonly total: number,
    private readonly hooks: BenchmarkHooks,
  ) {
    this.hooks.onProgress({ completed: 0, total, message: 'Preparing benchmark' });
  }

  step(message: string): void {
    this.completed++;
    this.hooks.onProgress({ completed: this.completed, total: this.total, message });
  }

  status(message: string): void {
    this.hooks.onProgress({ completed: this.completed, total: this.total, message });
  }
}

let sharedGpuPromise: Promise<SharedGpu> | undefined;
let nextGpuDeviceIdentity = 1;
const gpuDeviceIdentities = new WeakMap<GPUDevice, string>();
type LiteRtModule = typeof import('@litertjs/core', { with: { 'resolution-mode': 'import' } });
type LiteRtTensor = import('@litertjs/core', { with: { 'resolution-mode': 'import' } }).Tensor;

let liteRtLoadPromise: Promise<LiteRtModule> | undefined;
let liteRtInitializationMilliseconds: number | undefined;
let currentRequestGraphCapture = false;
let currentQueueSyncDiagnostic = false;
let currentWebGpuTrace = false;
let currentWallDecomposition = false;

const getGpuDeviceIdentity = (device: GPUDevice): string => {
  let identity = gpuDeviceIdentities.get(device);
  if (!identity) {
    identity = `gpu-device-${nextGpuDeviceIdentity++}`;
    gpuDeviceIdentities.set(device, identity);
  }
  return identity;
};

const getOrt = (): typeof OrtApi => {
  if (!window.ort) {
    throw new Error('ONNX Runtime Web did not load. Build the native WebGPU EP bundle and verify dist/ort.all.min.js.');
  }
  return window.ort;
};

const throwIfCancelled = (hooks: BenchmarkHooks): void => {
  if (hooks.isCancelled()) {
    throw new CancelledError();
  }
};

const createWallDecomposition = (
  apiTrace: WebGpuApiTrace,
  marker: WebGpuTraceMarker,
  totalStart: number,
  firstSubmitTimestamp: number | undefined,
  lastPreReadbackSubmitTimestamp: number | undefined,
  preReadbackSubmitCount: number,
  gpuDrainEnd: number,
  readbackEnd: number,
  traceSummary: Record<string, unknown>,
): NonNullable<RunInstrumentation['wallDecomposition']> => {
  const timeline = apiTrace.timelineSince(marker);
  const mapResolutionTimestamp = timeline.lastMapAsyncResolutionTimestamp;
  if (
    firstSubmitTimestamp === undefined ||
    lastPreReadbackSubmitTimestamp === undefined ||
    mapResolutionTimestamp === undefined
  ) {
    throw new Error(
      'Wall decomposition did not observe the required submit/map timeline. ' +
        `firstSubmit=${firstSubmitTimestamp}, lastSubmit=${lastPreReadbackSubmitTimestamp}, ` +
        `mapResolution=${mapResolutionTimestamp}.`,
    );
  }

  const counters = (traceSummary.counters ?? {}) as Record<string, number>;
  const encodeSetupMilliseconds = firstSubmitTimestamp - totalStart;
  const encodeSpanMilliseconds = lastPreReadbackSubmitTimestamp - firstSubmitTimestamp;
  const gpuDrainMilliseconds = gpuDrainEnd - lastPreReadbackSubmitTimestamp;
  const readbackMilliseconds = mapResolutionTimestamp - gpuDrainEnd;
  const copyOutPostMilliseconds = readbackEnd - mapResolutionTimestamp;
  const bucketSumMilliseconds =
    encodeSetupMilliseconds +
    encodeSpanMilliseconds +
    gpuDrainMilliseconds +
    readbackMilliseconds +
    copyOutPostMilliseconds;
  const wallMilliseconds = readbackEnd - totalStart;
  return {
    encodeSetupMilliseconds,
    encodeSpanMilliseconds,
    gpuDrainMilliseconds,
    readbackMilliseconds,
    copyOutPostMilliseconds,
    bucketSumMilliseconds,
    wallMilliseconds,
    bucketCoverageRatio: wallMilliseconds === 0 ? 1 : bucketSumMilliseconds / wallMilliseconds,
    preReadbackSubmitCount,
    totalSubmitCount: counters.queueSubmitCalls ?? 0,
    computePassCount: counters.beginComputePassCalls ?? counters.computePassCreations ?? 0,
    copyBufferToBufferCount: counters.copyBufferToBufferCalls ?? 0,
    mapAsyncCount: counters.bufferMapAsyncCalls ?? 0,
  };
};

const formatAssetSource = (selection: AssetSelection): string =>
  selection.file ? `local file: ${selection.file.name}` : `URL: ${selection.url.trim()}`;

const loadAsset = async (selection: AssetSelection, label: string, hooks: BenchmarkHooks): Promise<LoadedAsset> => {
  throwIfCancelled(hooks);
  hooks.onStatus(`Loading ${label} from ${formatAssetSource(selection)}`);

  let bytes: Uint8Array;
  let source: string;
  if (selection.file) {
    bytes = new Uint8Array(await selection.file.arrayBuffer());
    source = `file:${selection.file.name}`;
  } else {
    const url = selection.url.trim();
    if (!url) {
      throw new Error(`Select a local ${label} file or provide its URL.`);
    }
    let response: Response;
    try {
      response = await fetch(url);
    } catch (error) {
      throw new Error(
        `Could not fetch ${label} from ${url}. Use a same-origin URL or configure CORS. ${String(error)}`,
      );
    }
    if (!response.ok) {
      throw new Error(`Could not fetch ${label} from ${url}: HTTP ${response.status} ${response.statusText}.`);
    }
    bytes = new Uint8Array(await response.arrayBuffer());
    source = response.url || url;
  }

  throwIfCancelled(hooks);
  const sha256 = await sha256Hex(bytes);
  validateSha256(sha256, selection.expectedSha256, label);
  return { bytes, sha256, source, sizeBytes: bytes.byteLength };
};

const loadAssets = async (request: BenchmarkRequest, hooks: BenchmarkHooks): Promise<LoadedAssets> => {
  const input = await loadAsset(request.input, 'deterministic input NPY', hooks);
  const parseStart = performance.now();
  const parsedInput = parseFloat32Npy(input.bytes);
  const inputParsingMilliseconds = performance.now() - parseStart;
  requireShape(parsedInput.shape, ORT_INPUT_SHAPE, 'Deterministic input');

  const needsOrt = request.mode === 'comparison' || request.mode === 'ort';
  const needsLiteRt = request.mode === 'comparison' || request.mode === 'litert';
  const [onnx, tflite] = await Promise.all([
    needsOrt ? loadAsset(request.onnx, 'ONNX model', hooks) : Promise.resolve(undefined),
    needsLiteRt ? loadAsset(request.tflite, 'TFLite model', hooks) : Promise.resolve(undefined),
  ]);
  throwIfCancelled(hooks);

  const conversionStart = performance.now();
  const inputNhwc = needsLiteRt ? transposeNchwToNhwc(parsedInput.data, parsedInput.shape) : undefined;
  return {
    onnx,
    tflite,
    input,
    inputNchw: parsedInput.data,
    inputNhwc,
    inputLayoutConversionMilliseconds: needsLiteRt ? performance.now() - conversionStart : undefined,
    inputParsingMilliseconds,
  };
};

const getAdapterInfo = async (adapter: GPUAdapter): Promise<Record<string, unknown>> => {
  const adapterWithInfo = adapter as GPUAdapter & {
    readonly info?: GPUAdapterInfo;
    requestAdapterInfo?: () => Promise<GPUAdapterInfo>;
  };
  const info =
    adapterWithInfo.info ??
    (typeof adapterWithInfo.requestAdapterInfo === 'function' ? await adapterWithInfo.requestAdapterInfo() : undefined);
  if (!info) {
    return { available: false };
  }

  const rawInfo = info as unknown as Record<string, unknown>;
  const result: Record<string, unknown> = { available: true };
  for (const key of ['vendor', 'architecture', 'device', 'description', 'driver', 'backend']) {
    if (typeof rawInfo[key] === 'string' && rawInfo[key]) {
      result[key] = rawInfo[key];
    }
  }
  return result;
};

const getDeviceAdapterInfo = (device: GPUDevice): Record<string, unknown> => {
  const deviceWithInfo = device as GPUDevice & { readonly adapterInfo?: GPUAdapterInfo };
  if (!deviceWithInfo.adapterInfo) {
    return { available: false };
  }

  const rawInfo = deviceWithInfo.adapterInfo as unknown as Record<string, unknown>;
  const result: Record<string, unknown> = { available: true };
  for (const key of ['vendor', 'architecture', 'device', 'description', 'driver', 'backend']) {
    if (typeof rawInfo[key] === 'string' && rawInfo[key]) {
      result[key] = rawInfo[key];
    }
  }
  return result;
};

const snapshotLimits = (limits: GPUSupportedLimits): Record<string, number> => {
  const limitRecord = limits as unknown as Record<string, number>;
  const result: Record<string, number> = {};
  for (const name of [
    'maxBufferSize',
    'maxStorageBufferBindingSize',
    'maxComputeWorkgroupStorageSize',
    'maxComputeInvocationsPerWorkgroup',
    'maxComputeWorkgroupsPerDimension',
    'maxComputeWorkgroupSizeX',
    'maxComputeWorkgroupSizeY',
    'maxComputeWorkgroupSizeZ',
  ]) {
    if (typeof limitRecord[name] === 'number') {
      result[name] = limitRecord[name];
    }
  }
  return result;
};

const snapshotDevice = (device: GPUDevice): Record<string, unknown> => ({
  adapterInfo: getDeviceAdapterInfo(device),
  features: [...device.features].sort(),
  limits: snapshotLimits(device.limits),
});

const getBrowserDiagnostics = async (): Promise<Record<string, unknown>> => {
  const navigatorWithUserAgentData = navigator as Navigator & {
    readonly userAgentData?: {
      readonly brands: ReadonlyArray<{ readonly brand: string; readonly version: string }>;
      readonly mobile: boolean;
      readonly platform: string;
      getHighEntropyValues?: (hints: readonly string[]) => Promise<Record<string, unknown>>;
    };
  };
  const userAgentData = navigatorWithUserAgentData.userAgentData;
  let highEntropyValues: Record<string, unknown> | undefined;
  if (userAgentData?.getHighEntropyValues) {
    try {
      highEntropyValues = await userAgentData.getHighEntropyValues([
        'architecture',
        'bitness',
        'fullVersionList',
        'model',
        'platformVersion',
        'wow64',
      ]);
    } catch {
      highEntropyValues = undefined;
    }
  }

  return {
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    language: navigator.language,
    hardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated,
    secureContext: window.isSecureContext,
    userAgentData: userAgentData
      ? {
          brands: userAgentData.brands,
          mobile: userAgentData.mobile,
          platform: userAgentData.platform,
          highEntropyValues,
        }
      : undefined,
  };
};

const requestHighPerformanceAdapter = async (): Promise<GPUAdapter> => {
  if (!navigator.gpu) {
    throw new Error('WebGPU is unavailable. Use a current Chrome or Edge build on HTTPS or localhost.');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('navigator.gpu.requestAdapter({powerPreference:"high-performance"}) returned no adapter.');
  }
  return adapter;
};

const probeSharedGpu = async (gpu: SharedGpu): Promise<Record<string, unknown>> => {
  const { adapter } = gpu;
  return {
    adapterInfo: await getAdapterInfo(adapter),
    adapterFeatures: [...adapter.features].sort(),
    adapterLimits: snapshotLimits(adapter.limits),
    powerPreference: 'high-performance',
    note: 'This exact GPUAdapter created the one GPUDevice supplied to both runtimes.',
  };
};

const createCallerSharedGpu = async (): Promise<SharedGpu> => {
  const adapter = await requestHighPerformanceAdapter();
  const requiredFeatures = (['timestamp-query', 'shader-f16', 'subgroups'] as GPUFeatureName[]).filter((feature) =>
    adapter.features.has(feature),
  );
  const requiredLimits = {
    maxBindGroups: adapter.limits.maxBindGroups,
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
  };
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });

  const diagnostics = {
    creation: 'caller-created high-performance device supplied to native ORT WebGPU EP and LiteRT.js',
    adapterInfo: await getAdapterInfo(adapter),
    adapterFeatures: [...adapter.features].sort(),
    adapterLimits: snapshotLimits(adapter.limits),
    requestedDeviceFeatures: requiredFeatures,
    requestedDeviceLimits: requiredLimits,
    device: snapshotDevice(device),
    powerPreference: 'high-performance',
    sameAdapterProof: 'One GPUAdapter object created the shared device.',
    sameDeviceProof: 'The exact GPUDevice reference is passed to both public runtime APIs.',
  };

  return { adapter, device, diagnostics, origin: 'caller-created-shared' };
};

const ensureSharedGpu = async (): Promise<SharedGpu> => {
  if (!sharedGpuPromise) {
    sharedGpuPromise = createCallerSharedGpu().catch((error: unknown) => {
      sharedGpuPromise = undefined;
      throw error;
    });
  }
  return sharedGpuPromise;
};

const ensureLiteRtLoaded = async (): Promise<LiteRtModule> => {
  if (!liteRtLoadPromise) {
    liteRtLoadPromise = (async () => {
      const liteRt = await import('@litertjs/core');
      const start = performance.now();
      await liteRt.loadLiteRt('./dist/litert-wasm/', { threads: false, jspi: false });
      liteRtInitializationMilliseconds = performance.now() - start;
      return liteRt;
    })();
  }
  return liteRtLoadPromise;
};

const validateModelMetadata = (
  metadata: readonly OrtApi.InferenceSession.ValueMetadata[],
  expectedShape: readonly number[],
  expectedType: OrtApi.Tensor.Type,
  label: string,
): void => {
  if (metadata.length !== 1 || !metadata[0].isTensor) {
    throw new Error(`${label} must contain exactly one tensor.`);
  }
  if (metadata[0].type !== expectedType) {
    throw new Error(`${label} must use ${expectedType}, got ${metadata[0].type}.`);
  }
  const shape = metadata[0].shape;
  if (!shape.every((dimension): dimension is number => typeof dimension === 'number')) {
    throw new Error(`${label} must have a static shape, got [${shape.join(',')}].`);
  }
  requireShape(shape, expectedShape, label);
};

const createOrtRunner = async (assets: LoadedAssets, inputBoundary: InputBoundary): Promise<ArmRunner> => {
  if (!assets.onnx) {
    throw new Error('The ONNX model was not loaded.');
  }
  const ort = getOrt();
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  const verbose = new URLSearchParams(window.location.search).get('verbose') === '1';
  ort.env.debug = verbose;
  ort.env.logLevel = verbose ? 'verbose' : 'warning';
  const { device } = await ensureSharedGpu();

  const webGpuExecutionProvider: OrtApi.InferenceSession.WebGpuExecutionProviderOption = {
    name: 'webgpu',
    device,
    validationMode: 'wgpuOnly',
  };
  const options: OrtApi.InferenceSession.SessionOptions = {
    executionProviders: [webGpuExecutionProvider],
    graphOptimizationLevel: 'all',
    enableGraphCapture: currentRequestGraphCapture,
    preferredOutputLocation: 'gpu-buffer',
    extra: { session: { disable_cpu_ep_fallback: '1' } },
  };

  let apiTrace = currentWebGpuTrace ? new WebGpuApiTrace(device) : undefined;
  apiTrace?.install();
  apiTrace?.setPhase('compile');
  const compileTraceMarker = apiTrace?.mark();
  const compileStart = performance.now();
  let session: OrtApi.InferenceSession;
  try {
    session = await ort.InferenceSession.create(assets.onnx.bytes, options);
  } catch (error) {
    apiTrace?.restore();
    throw error;
  }
  const compileMilliseconds = performance.now() - compileStart;
  const compileTrace =
    apiTrace && compileTraceMarker
      ? await apiTrace.summarizeSince(compileTraceMarker, {
          includeDispatchSequence: true,
          includeShaderMetadata: true,
        })
      : undefined;
  try {
    validateModelMetadata(session.inputMetadata, ORT_INPUT_SHAPE, 'float32', 'ONNX model input');
    validateModelMetadata(session.outputMetadata, CANONICAL_OUTPUT_SHAPE, 'float32', 'ONNX model output');
  } catch (error) {
    await session.release();
    apiTrace?.restore();
    throw error;
  }

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  let inputTensor: OrtApi.Tensor;
  apiTrace?.setPhase('input-setup');
  const inputSetupTraceMarker = apiTrace?.mark();
  try {
    if (inputBoundary === 'gpu-resident') {
      const gpuBuffer = device.createBuffer({
        size: assets.inputNchw.byteLength,
        // eslint-disable-next-line no-bitwise
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
      });
      device.queue.writeBuffer(
        gpuBuffer,
        0,
        assets.inputNchw.buffer,
        assets.inputNchw.byteOffset,
        assets.inputNchw.byteLength,
      );
      await device.queue.onSubmittedWorkDone();
      inputTensor = ort.Tensor.fromGpuBuffer(gpuBuffer, {
        dataType: 'float32',
        dims: ORT_INPUT_SHAPE,
        dispose: () => gpuBuffer.destroy(),
      });
    } else {
      inputTensor = new ort.Tensor('float32', assets.inputNchw, ORT_INPUT_SHAPE);
    }
  } catch (error) {
    await session.release();
    apiTrace?.restore();
    throw error;
  }
  const inputSetupTrace =
    apiTrace && inputSetupTraceMarker
      ? await apiTrace.summarizeSince(inputSetupTraceMarker, { includeDispatchSequence: true })
      : undefined;

  let wallDecompositionEnabled = false;
  const enableWallDecomposition = (): void => {
    if (!apiTrace) {
      apiTrace = new WebGpuApiTrace(device);
      apiTrace.install();
    }
    wallDecompositionEnabled = true;
  };

  let includeDispatchSequence = true;
  const runOnce = async (readbackOutput = true): Promise<InferenceOutput> => {
    apiTrace?.setPhase('inference');
    const inferenceTraceMarker = apiTrace?.mark();
    const totalStart = performance.now();
    const schedulingStart = totalStart;
    const outputs = await session.run({ [inputName]: inputTensor });
    const schedulingEnd = performance.now();
    let gpuCompletionMilliseconds: number | undefined;
    let firstSubmitTimestamp: number | undefined;
    let lastPreReadbackSubmitTimestamp: number | undefined;
    let preReadbackSubmitCount = 0;
    let gpuDrainEnd: number | undefined;
    if (wallDecompositionEnabled && apiTrace && inferenceTraceMarker) {
      const preReadbackTimeline = apiTrace.timelineSince(inferenceTraceMarker);
      firstSubmitTimestamp = preReadbackTimeline.firstSubmitTimestamp;
      lastPreReadbackSubmitTimestamp = preReadbackTimeline.lastSubmitTimestamp;
      preReadbackSubmitCount = preReadbackTimeline.submitCount;
      const gpuCompletionStart = performance.now();
      await device.queue.onSubmittedWorkDone();
      gpuDrainEnd = performance.now();
      gpuCompletionMilliseconds = gpuDrainEnd - gpuCompletionStart;
    } else if (currentQueueSyncDiagnostic) {
      const gpuCompletionStart = performance.now();
      await device.queue.onSubmittedWorkDone();
      gpuCompletionMilliseconds = performance.now() - gpuCompletionStart;
    }
    const output = outputs[outputName];
    if (!(output instanceof ort.Tensor)) {
      throw new Error(`ORT output "${outputName}" is not a tensor.`);
    }
    const shape = [...output.dims];
    if (output.location !== 'gpu-buffer') {
      output.dispose();
      throw new Error(`ORT output "${outputName}" is at ${output.location}; expected gpu-buffer before readback.`);
    }
    if (!readbackOutput) {
      if (gpuDrainEnd === undefined) {
        const gpuCompletionStart = performance.now();
        await device.queue.onSubmittedWorkDone();
        gpuDrainEnd = performance.now();
        gpuCompletionMilliseconds = gpuDrainEnd - gpuCompletionStart;
      }
      const webGpuApiTrace =
        apiTrace && inferenceTraceMarker
          ? await apiTrace.summarizeSince(inferenceTraceMarker, {
              includeDispatchSequence,
              includeShaderMetadata: includeDispatchSequence,
            })
          : undefined;
      includeDispatchSequence = false;
      output.dispose();
      return {
        data: new Float32Array(),
        shape,
        instrumentation: {
          schedulingMilliseconds: schedulingEnd - schedulingStart,
          gpuCompletionMilliseconds,
          readbackMilliseconds: 0,
          totalMilliseconds: gpuDrainEnd - totalStart,
          webGpuApiTrace,
        },
      };
    }
    try {
      const readbackStart = performance.now();
      const data = await output.getData(true);
      const readbackEnd = performance.now();
      if (!(data instanceof Float32Array)) {
        throw new Error(`ORT output "${outputName}" uses ${output.type}; expected float32.`);
      }
      const webGpuApiTrace =
        apiTrace && inferenceTraceMarker
          ? await apiTrace.summarizeSince(inferenceTraceMarker, {
              includeDispatchSequence,
              includeShaderMetadata: includeDispatchSequence,
            })
          : undefined;
      const wallDecomposition =
        wallDecompositionEnabled && apiTrace && inferenceTraceMarker && webGpuApiTrace && gpuDrainEnd !== undefined
          ? createWallDecomposition(
              apiTrace,
              inferenceTraceMarker,
              totalStart,
              firstSubmitTimestamp,
              lastPreReadbackSubmitTimestamp,
              preReadbackSubmitCount,
              gpuDrainEnd,
              readbackEnd,
              webGpuApiTrace,
            )
          : undefined;
      includeDispatchSequence = false;
      return {
        data,
        shape,
        instrumentation: currentQueueSyncDiagnostic || currentWebGpuTrace || wallDecomposition
          ? {
              schedulingMilliseconds: schedulingEnd - schedulingStart,
              gpuCompletionMilliseconds,
              readbackMilliseconds: readbackEnd - readbackStart,
              totalMilliseconds: readbackEnd - totalStart,
              webGpuApiTrace,
              wallDecomposition,
            }
          : undefined,
      };
    } finally {
      output.dispose();
    }
  };

  return {
    arm: 'ort',
    compileMilliseconds,
    proof: {
      accelerator: 'webgpu',
      deviceApiProof:
        'The exact caller-created GPUDevice is passed to the native C++ WebGPU EP and LiteRT.js; ' +
        'browser WebGPU validation remains enabled with validationMode=wgpuOnly.',
      deviceIdentity: getGpuDeviceIdentity(device),
      deviceReferenceVerified: true,
      deviceSnapshot: snapshotDevice(device),
      fallbackDisabled: true,
      inputShape: [...ORT_INPUT_SHAPE],
      outputShape: [...CANONICAL_OUTPUT_SHAPE],
      outputLocation: 'gpu-buffer requested; verified before every getData() readback',
      precision: 'FP32 ONNX model, input, output, and GPU buffers',
      webGpuApiTrace: apiTrace ? { compile: compileTrace, inputSetup: inputSetupTrace } : undefined,
    },
    enableWallDecomposition,
    runOnce,
    async dispose(): Promise<void> {
      apiTrace?.setPhase('dispose');
      try {
        await device.queue.onSubmittedWorkDone();
        inputTensor.dispose();
        await session.release();
        await device.queue.onSubmittedWorkDone();
      } finally {
        apiTrace?.restore();
      }
    },
  };
};

const isSupportedOutputShape = (shape: readonly number[]): boolean =>
  [CANONICAL_OUTPUT_SHAPE, TRANSPOSED_OUTPUT_SHAPE].some(
    (expected) => expected.length === shape.length && expected.every((value, index) => value === shape[index]),
  );

const instrumentationEnabled = (): boolean =>
  currentQueueSyncDiagnostic || currentWebGpuTrace || currentWallDecomposition;

const createLiteRtRunner = async (assets: LoadedAssets, inputBoundary: InputBoundary): Promise<ArmRunner> => {
  if (!assets.tflite || !assets.inputNhwc) {
    throw new Error('The TFLite model or its NHWC input was not loaded.');
  }
  const { device } = await ensureSharedGpu();
  const liteRt = await ensureLiteRtLoaded();
  let apiTrace = currentWebGpuTrace ? new WebGpuApiTrace(device) : undefined;
  apiTrace?.install();
  apiTrace?.setPhase('compile');
  const compileTraceMarker = apiTrace?.mark();
  let environment: Awaited<ReturnType<typeof liteRt.Environment.create>>;
  try {
    environment = await liteRt.Environment.create({ webGpuDevice: device });
  } catch (error) {
    apiTrace?.restore();
    throw error;
  }
  const compileStart = performance.now();
  let model;
  try {
    model = await liteRt.loadAndCompile(assets.tflite.bytes, {
      accelerator: 'webgpu',
      environment,
      gpuOptions: { precision: 'fp32' },
    });
  } catch (error) {
    environment.delete();
    apiTrace?.restore();
    throw error;
  }
  const compileMilliseconds = performance.now() - compileStart;
  const compileTrace =
    apiTrace && compileTraceMarker
      ? await apiTrace.summarizeSince(compileTraceMarker, {
          includeDispatchSequence: true,
          includeShaderMetadata: true,
        })
      : undefined;
  const failSetup = (message: string): never => {
    model.delete();
    environment.delete();
    apiTrace?.restore();
    throw new Error(message);
  };

  if (model.options.accelerator !== 'webgpu') {
    failSetup(
      `LiteRT.js requested WebGPU but compiled with ${model.options.accelerator}; refusing a CPU fallback result.`,
    );
  }
  if (!model.isFullyAccelerated) {
    failSetup('LiteRT.js reports isFullyAccelerated=false; refusing a hybrid WebGPU/WASM result.');
  }
  if (model.options.gpuOptions.precision !== 'fp32') {
    failSetup(`LiteRT.js did not retain the FP32 request: ${model.options.gpuOptions.precision}.`);
  }
  const liteRtDevice =
    environment.webGpuDevice ?? failSetup('LiteRT.js Environment does not expose its WebGPU device.');
  if (liteRtDevice !== device) {
    failSetup('LiteRT.js Environment does not reference the shared GPUDevice.');
  }

  let inputShape: number[];
  let outputShape: number[];
  let inputTensor: LiteRtTensor;
  apiTrace?.setPhase('input-setup');
  const inputSetupTraceMarker = apiTrace?.mark();
  try {
    const inputDetails = model.getInputDetails();
    const outputDetails = model.getOutputDetails();
    if (inputDetails.length !== 1 || inputDetails[0].dtype !== 'float32') {
      throw new Error('TFLite model must contain exactly one float32 input.');
    }
    if (outputDetails.length !== 1 || outputDetails[0].dtype !== 'float32') {
      throw new Error('TFLite model must contain exactly one float32 output.');
    }
    inputShape = [...inputDetails[0].shape];
    requireShape(inputShape, LITERT_INPUT_SHAPE, 'TFLite model input');
    outputShape = [...outputDetails[0].shape];
    if (!isSupportedOutputShape(outputShape)) {
      throw new Error(
        `TFLite model output must be [${CANONICAL_OUTPUT_SHAPE.join(',')}] or ` +
          `[${TRANSPOSED_OUTPUT_SHAPE.join(',')}], got [${outputShape.join(',')}].`,
      );
    }

    if (inputBoundary === 'gpu-resident') {
      const hostTensor = new liteRt.Tensor(assets.inputNhwc, inputShape, environment);
      try {
        inputTensor = await hostTensor.moveTo('webgpu', { environment });
      } finally {
        if (!hostTensor.deleted) {
          hostTensor.delete();
        }
      }
    } else {
      inputTensor = new liteRt.Tensor(assets.inputNhwc, inputShape, environment);
    }
  } catch (error) {
    model.delete();
    environment.delete();
    apiTrace?.restore();
    throw error;
  }
  const inputSetupTrace =
    apiTrace && inputSetupTraceMarker
      ? await apiTrace.summarizeSince(inputSetupTraceMarker, { includeDispatchSequence: true })
      : undefined;

  let wallDecompositionEnabled = false;
  const enableWallDecomposition = (): void => {
    if (!apiTrace) {
      apiTrace = new WebGpuApiTrace(device);
      apiTrace.install();
    }
    wallDecompositionEnabled = true;
  };

  let includeDispatchSequence = true;
  const runOnce = async (readbackOutput = true): Promise<InferenceOutput> => {
    apiTrace?.setPhase('inference');
    const inferenceTraceMarker = apiTrace?.mark();
    const totalStart = performance.now();
    const schedulingStart = totalStart;
    const outputs = await model.run(inputTensor);
    const schedulingEnd = performance.now();
    let gpuCompletionMilliseconds: number | undefined;
    let firstSubmitTimestamp: number | undefined;
    let lastPreReadbackSubmitTimestamp: number | undefined;
    let preReadbackSubmitCount = 0;
    let gpuDrainEnd: number | undefined;
    if (wallDecompositionEnabled && apiTrace && inferenceTraceMarker) {
      const preReadbackTimeline = apiTrace.timelineSince(inferenceTraceMarker);
      firstSubmitTimestamp = preReadbackTimeline.firstSubmitTimestamp;
      lastPreReadbackSubmitTimestamp = preReadbackTimeline.lastSubmitTimestamp;
      preReadbackSubmitCount = preReadbackTimeline.submitCount;
      const gpuCompletionStart = performance.now();
      await device.queue.onSubmittedWorkDone();
      gpuDrainEnd = performance.now();
      gpuCompletionMilliseconds = gpuDrainEnd - gpuCompletionStart;
    } else if (currentQueueSyncDiagnostic) {
      const gpuCompletionStart = performance.now();
      await device.queue.onSubmittedWorkDone();
      gpuCompletionMilliseconds = performance.now() - gpuCompletionStart;
    }
    if (outputs.length !== 1) {
      for (const output of outputs) {
        output.delete();
      }
      throw new Error(`LiteRT.js returned ${outputs.length} outputs; expected one.`);
    }

    const output = outputs[0];
    if (!readbackOutput) {
      if (gpuDrainEnd === undefined) {
        const gpuCompletionStart = performance.now();
        await device.queue.onSubmittedWorkDone();
        gpuDrainEnd = performance.now();
        gpuCompletionMilliseconds = gpuDrainEnd - gpuCompletionStart;
      }
      const webGpuApiTrace =
        apiTrace && inferenceTraceMarker
          ? await apiTrace.summarizeSince(inferenceTraceMarker, {
              includeDispatchSequence,
              includeShaderMetadata: includeDispatchSequence,
            })
          : undefined;
      includeDispatchSequence = false;
      output.delete();
      return {
        data: new Float32Array(),
        shape: outputShape,
        instrumentation: {
          schedulingMilliseconds: schedulingEnd - schedulingStart,
          gpuCompletionMilliseconds,
          readbackMilliseconds: 0,
          totalMilliseconds: gpuDrainEnd - totalStart,
          webGpuApiTrace,
        },
      };
    }
    try {
      const readbackStart = performance.now();
      const data = await output.data();
      const readbackEnd = performance.now();
      if (!(data instanceof Float32Array)) {
        throw new Error(`LiteRT.js output uses ${output.type.dtype}; expected float32.`);
      }
      const webGpuApiTrace =
        apiTrace && inferenceTraceMarker
          ? await apiTrace.summarizeSince(inferenceTraceMarker, {
              includeDispatchSequence,
              includeShaderMetadata: includeDispatchSequence,
            })
          : undefined;
      const wallDecomposition =
        wallDecompositionEnabled && apiTrace && inferenceTraceMarker && webGpuApiTrace && gpuDrainEnd !== undefined
          ? createWallDecomposition(
              apiTrace,
              inferenceTraceMarker,
              totalStart,
              firstSubmitTimestamp,
              lastPreReadbackSubmitTimestamp,
              preReadbackSubmitCount,
              gpuDrainEnd,
              readbackEnd,
              webGpuApiTrace,
            )
          : undefined;
      includeDispatchSequence = false;
      return {
        data,
        shape: outputShape,
        instrumentation: instrumentationEnabled()
          ? {
              schedulingMilliseconds: schedulingEnd - schedulingStart,
              gpuCompletionMilliseconds,
              readbackMilliseconds: readbackEnd - readbackStart,
              totalMilliseconds: readbackEnd - totalStart,
              webGpuApiTrace,
              wallDecomposition,
            }
          : undefined,
      };
    } finally {
      output.delete();
    }
  };

  return {
    arm: 'litert',
    compileMilliseconds,
    proof: {
      accelerator: model.options.accelerator,
      deviceApiProof: 'new Environment({webGpuDevice: sharedDevice})',
      deviceIdentity: getGpuDeviceIdentity(liteRtDevice),
      deviceReferenceVerified: liteRtDevice === device,
      deviceSnapshot: snapshotDevice(device),
      fullyAccelerated: model.isFullyAccelerated,
      inputShape,
      outputShape,
      outputLocation: 'WebGPU tensor followed by Tensor.data() CPU readback',
      precision: 'gpuOptions.precision=fp32; request echoed by model options (no post-hoc precision API)',
      webGpuApiTrace: apiTrace ? { compile: compileTrace, inputSetup: inputSetupTrace } : undefined,
    },
    enableWallDecomposition,
    runOnce,
    async dispose(): Promise<void> {
      apiTrace?.setPhase('dispose');
      try {
        await device.queue.onSubmittedWorkDone();
        inputTensor.delete();
        model.delete();
        environment.delete();
        await device.queue.onSubmittedWorkDone();
      } finally {
        apiTrace?.restore();
      }
    },
  };
};

const createRunner = async (arm: ArmName, assets: LoadedAssets, inputBoundary: InputBoundary): Promise<ArmRunner> =>
  arm === 'ort' ? createOrtRunner(assets, inputBoundary) : createLiteRtRunner(assets, inputBoundary);

const timedRun = async (
  runner: ArmRunner,
  readbackOutput = true,
): Promise<{ readonly elapsed: number; readonly output: InferenceOutput }> => {
  const start = performance.now();
  const output = await runner.runOnce(readbackOutput);
  return { elapsed: performance.now() - start, output };
};

const validateArmOutput = async (
  arm: ArmName,
  assets: LoadedAssets,
  inputBoundary: InputBoundary,
  hooks: BenchmarkHooks,
  progress: ProgressTracker,
): Promise<{ readonly result: CorrectnessArmResult; readonly output: Float32Array }> => {
  throwIfCancelled(hooks);
  progress.status(`Compiling ${arm === 'ort' ? 'ONNX Runtime' : 'LiteRT.js'} for correctness`);
  const runner = await createRunner(arm, assets, inputBoundary);
  progress.step(`${arm} correctness model compiled`);
  try {
    throwIfCancelled(hooks);
    const inference = await timedRun(runner);
    const normalized = normalizeOutput(inference.output.data, inference.output.shape);
    const usesKnownTfliteBoxNormalization = arm === 'litert' && assets.tflite?.sha256 === EXPECTED_TFLITE_SHA256;
    const semanticallyNormalizedData = usesKnownTfliteBoxNormalization
      ? scaleYoloBoxChannels(normalized.data)
      : normalized.data;
    progress.step(`${arm} correctness output read back`);
    return {
      result: {
        compileMilliseconds: runner.compileMilliseconds,
        inferenceMilliseconds: inference.elapsed,
        sourceShape: normalized.sourceShape,
        normalizedShape: normalized.normalizedShape,
        normalization:
          normalized.normalization + (usesKnownTfliteBoxNormalization ? '; multiply box channels [0,4) by 640' : ''),
        runtimeProof: {
          ...runner.proof,
          outputLocation: arm === 'ort' ? 'gpu-buffer then getData()' : runner.proof.outputLocation,
        },
      },
      output: new Float32Array(semanticallyNormalizedData),
    };
  } finally {
    await runner.dispose();
  }
};

const runArmRound = async (
  arm: ArmName,
  round: number,
  orderInRound: number,
  assets: LoadedAssets,
  request: BenchmarkRequest,
  hooks: BenchmarkHooks,
  progress: ProgressTracker,
): Promise<RoundResult> => {
  throwIfCancelled(hooks);
  const displayName = arm === 'ort' ? 'ONNX Runtime' : 'LiteRT.js';
  progress.status(`Round ${round}: compiling ${displayName}`);
  const runner = await createRunner(arm, assets, request.inputBoundary);
  progress.step(`Round ${round}: ${displayName} compiled`);
  const readbackOutput = !request.gpuResidentOutput;

  try {
    throwIfCancelled(hooks);
    if (request.stabilizationMilliseconds > 0) {
      progress.status(
        `Round ${round}: stabilizing ${displayName} for ${request.stabilizationMilliseconds} ms before inference`,
      );
      await new Promise<void>((resolve) => {
        setTimeout(resolve, request.stabilizationMilliseconds);
      });
      throwIfCancelled(hooks);
    }
    const firstInference = await timedRun(runner, readbackOutput);
    progress.step(`Round ${round}: ${displayName} first inference complete`);

    for (let index = 0; index < request.warmups; index++) {
      throwIfCancelled(hooks);
      await runner.runOnce(readbackOutput);
      progress.step(`Round ${round}: ${displayName} warmup ${index + 1}/${request.warmups}`);
    }

    const samples: number[] = [];
    const instrumentationSamples: RunInstrumentation[] = [];
    for (let index = 0; index < request.iterations; index++) {
      throwIfCancelled(hooks);
      const inference = await timedRun(runner, readbackOutput);
      samples.push(inference.elapsed);
      if (inference.output.instrumentation) {
        instrumentationSamples.push(inference.output.instrumentation);
      }
      progress.step(`Round ${round}: ${displayName} sample ${index + 1}/${request.iterations}`);
    }

    let wallDecomposition: RoundResult['wallDecomposition'];
    if (request.wallDecomposition) {
      if (!runner.enableWallDecomposition) {
        throw new Error(`${displayName} does not support wall decomposition.`);
      }
      runner.enableWallDecomposition();
      const decompositionSamples: number[] = [];
      const decompositionInstrumentation: RunInstrumentation[] = [];
      for (let index = 0; index < request.iterations; index++) {
        throwIfCancelled(hooks);
        const inference = await timedRun(runner, readbackOutput);
        decompositionSamples.push(inference.elapsed);
        if (!inference.output.instrumentation?.wallDecomposition) {
          throw new Error(`${displayName} wall decomposition sample did not include bucket timing.`);
        }
        decompositionInstrumentation.push(inference.output.instrumentation);
        progress.step(`Round ${round}: ${displayName} decomposition sample ${index + 1}/${request.iterations}`);
      }
      wallDecomposition = {
        samplesMilliseconds: decompositionSamples,
        statistics: calculateStatistics(decompositionSamples),
        instrumentation: decompositionInstrumentation,
      };
    }

    return {
      arm,
      round,
      orderInRound,
      compileMilliseconds: runner.compileMilliseconds,
      firstInferenceMilliseconds: firstInference.elapsed,
      samplesMilliseconds: samples,
      statistics: calculateStatistics(samples),
      instrumentation: firstInference.output.instrumentation
        ? {
            firstInference: firstInference.output.instrumentation,
            timedSamples: instrumentationSamples,
          }
        : undefined,
      wallDecomposition,
      runtimeProof: {
        ...runner.proof,
        outputLocation: readbackOutput
          ? arm === 'ort'
            ? 'gpu-buffer then getData()'
            : runner.proof.outputLocation
          : 'GPU-resident output followed by queue completion; no CPU readback',
      },
    };
  } finally {
    await runner.dispose();
  }
};

const aggregateRounds = (rounds: readonly RoundResult[], arm: ArmName): Statistics | undefined => {
  const samples = rounds.filter((round) => round.arm === arm).flatMap((round) => round.samplesMilliseconds);
  return samples.length > 0 ? calculateStatistics(samples) : undefined;
};

const validateRequest = (request: BenchmarkRequest): void => {
  if (!Number.isSafeInteger(request.warmups) || request.warmups < 0) {
    throw new Error('Warmups must be a non-negative integer.');
  }
  if (!Number.isSafeInteger(request.iterations) || request.iterations < 1) {
    throw new Error('Iterations must be a positive integer.');
  }
  if (!Number.isSafeInteger(request.rounds) || request.rounds < 1) {
    throw new Error('Rounds must be a positive integer.');
  }
  if (!Number.isSafeInteger(request.stabilizationMilliseconds) || request.stabilizationMilliseconds < 0) {
    throw new Error('Stabilization interval must be a non-negative integer.');
  }
  if (request.gpuResidentOutput && request.wallDecomposition) {
    throw new Error('GPU-resident output timing and readback wall decomposition cannot be enabled together.');
  }
};

const summarizeAsset = (asset: LoadedAsset | undefined): Record<string, unknown> | undefined =>
  asset
    ? {
        sha256: asset.sha256,
        source: asset.source,
        sizeBytes: asset.sizeBytes,
      }
    : undefined;

const hasCrossRuntimeSharedDeviceProof = (rounds: readonly RoundResult[]): boolean => {
  const arms = new Set(rounds.map((round) => round.arm));
  const deviceIdentities = new Set(rounds.map((round) => round.runtimeProof.deviceIdentity));
  return (
    arms.has('ort') &&
    arms.has('litert') &&
    deviceIdentities.size === 1 &&
    rounds.every((round) => round.runtimeProof.deviceReferenceVerified)
  );
};

const getQualification = (
  request: BenchmarkRequest,
  assets: LoadedAssets,
  rounds: readonly RoundResult[],
  correctness?: { readonly comparison: ReturnType<typeof compareOutputs> },
): { readonly status: 'presentation-qualified' | 'development-only'; readonly reasons: readonly string[] } => {
  const reasons: string[] = [];
  if (request.mode !== 'comparison') {
    reasons.push('Only a matched two-runtime comparison can be presentation-qualified.');
  }
  if (
    request.queueSyncDiagnostic ||
    request.webGpuTrace ||
    request.wallDecomposition ||
    request.gpuResidentOutput
  ) {
    reasons.push('Instrumented runs are development-only and cannot be presentation-qualified.');
  }
  if (assets.onnx?.sha256 !== EXPECTED_ONNX_SHA256) {
    reasons.push('ONNX model hash is not the required launch-era raw model hash.');
  }
  if (assets.tflite?.sha256 !== EXPECTED_TFLITE_SHA256) {
    reasons.push('TFLite model hash is not the required raw FP32 model hash.');
  }
  if (assets.input.sha256 !== EXPECTED_INPUT_SHA256) {
    reasons.push('Input NPY hash is not the known deterministic input hash.');
  }
  if (request.inputBoundary !== 'gpu-resident') {
    reasons.push('Per-run host upload is a secondary boundary, not the qualified persistent-GPU-input boundary.');
  }
  if (request.warmups < 20 || request.iterations < 100 || request.rounds < 5) {
    reasons.push('Qualified results require at least 5 rounds x (20 warmups + 100 timed samples).');
  }
  if (request.order !== 'randomized') {
    reasons.push('Qualified results require randomized runtime starting order.');
  }
  if (request.stabilizationMilliseconds < 1000) {
    reasons.push('Qualified results require at least 1000 ms stabilization before each arm.');
  }
  if (!correctness) {
    reasons.push('Cross-runtime correctness was not checked before timing.');
  } else {
    if (correctness.comparison.nonFiniteCount !== 0) {
      reasons.push('Cross-runtime output comparison contains non-finite values.');
    }
    if (correctness.comparison.maxAbsoluteError > QUALIFIED_MAX_ABSOLUTE_ERROR) {
      reasons.push(
        `Max absolute error exceeds ${QUALIFIED_MAX_ABSOLUTE_ERROR}: ` + `${correctness.comparison.maxAbsoluteError}.`,
      );
    }
    if (correctness.comparison.p99AbsoluteError > QUALIFIED_P99_ABSOLUTE_ERROR) {
      reasons.push(
        `P99 absolute error exceeds ${QUALIFIED_P99_ABSOLUTE_ERROR}: ` + `${correctness.comparison.p99AbsoluteError}.`,
      );
    }
  }
  if (!hasCrossRuntimeSharedDeviceProof(rounds)) {
    reasons.push('ORT and LiteRT.js did not record the same GPUDevice object identity.');
  }
  if (rounds.some((round) => round.arm === 'ort' && !round.runtimeProof.fallbackDisabled)) {
    reasons.push('ORT CPU fallback was not disabled.');
  }
  if (rounds.some((round) => round.arm === 'litert' && !round.runtimeProof.fullyAccelerated)) {
    reasons.push('LiteRT.js did not report full WebGPU acceleration.');
  }

  return { status: reasons.length === 0 ? 'presentation-qualified' : 'development-only', reasons };
};

export const runBenchmark = async (
  request: BenchmarkRequest,
  hooks: BenchmarkHooks,
): Promise<Record<string, unknown>> => {
  validateRequest(request);
  throwIfCancelled(hooks);
  const ort = getOrt();
  const wasmArtifactPath = new URLSearchParams(window.location.search).get('wasmPath');
  if (wasmArtifactPath) {
    ort.env.wasm.wasmPaths = wasmArtifactPath;
  }
  currentRequestGraphCapture = request.graphCapture;
  currentQueueSyncDiagnostic = request.queueSyncDiagnostic;
  currentWebGpuTrace = request.webGpuTrace;
  currentWallDecomposition = request.wallDecomposition;

  const comparisonArmCount = request.mode === 'comparison' ? 2 : 0;
  const timedArmCount = request.mode === 'comparison' ? 2 : 1;
  const totalProgress =
    comparisonArmCount * 2 +
    request.rounds *
      timedArmCount *
      (2 + request.warmups + request.iterations + (request.wallDecomposition ? request.iterations : 0));
  const progress = new ProgressTracker(totalProgress, hooks);

  const assets = await loadAssets(request, hooks);
  progress.status('Creating the shared high-performance adapter and GPUDevice before runtime initialization');
  const gpu = await ensureSharedGpu();
  const adapterProbe = await probeSharedGpu(gpu);
  const browser = await getBrowserDiagnostics();
  throwIfCancelled(hooks);

  let correctness:
    | {
        readonly ort: CorrectnessArmResult;
        readonly litert: CorrectnessArmResult;
        readonly comparison: ReturnType<typeof compareOutputs>;
        readonly gate: {
          readonly passed: boolean;
          readonly maxAbsoluteErrorTolerance: number;
          readonly p99AbsoluteErrorTolerance: number;
        };
      }
    | undefined;

  if (request.mode === 'comparison') {
    const ortCorrectness = await validateArmOutput('ort', assets, request.inputBoundary, hooks, progress);
    const liteRtCorrectness = await validateArmOutput('litert', assets, request.inputBoundary, hooks, progress);
    const comparison = compareOutputs(ortCorrectness.output, liteRtCorrectness.output);
    const passed =
      comparison.nonFiniteCount === 0 &&
      comparison.maxAbsoluteError <= QUALIFIED_MAX_ABSOLUTE_ERROR &&
      comparison.p99AbsoluteError <= QUALIFIED_P99_ABSOLUTE_ERROR;
    correctness = {
      ort: ortCorrectness.result,
      litert: liteRtCorrectness.result,
      comparison,
      gate: {
        passed,
        maxAbsoluteErrorTolerance: QUALIFIED_MAX_ABSOLUTE_ERROR,
        p99AbsoluteErrorTolerance: QUALIFIED_P99_ABSOLUTE_ERROR,
      },
    };
    if (!passed) {
      throw new Error(
        'Correctness gate failed before timing: ' +
          `max abs=${comparison.maxAbsoluteError}, p99 abs=${comparison.p99AbsoluteError}, ` +
          `non-finite=${comparison.nonFiniteCount}.`,
      );
    }
  }

  const rounds: RoundResult[] = [];
  const roundOrders: Array<{ readonly round: number; readonly order: readonly ArmName[] }> = [];
  const randomizedOrderValue = Math.random();
  for (let roundIndex = 0; roundIndex < request.rounds; roundIndex++) {
    throwIfCancelled(hooks);
    const fullOrder =
      request.mode === 'comparison'
        ? getRoundOrder(roundIndex, request.order, randomizedOrderValue)
        : ([request.mode] as const);
    roundOrders.push({ round: roundIndex + 1, order: fullOrder });
    for (let orderIndex = 0; orderIndex < fullOrder.length; orderIndex++) {
      rounds.push(
        await runArmRound(fullOrder[orderIndex], roundIndex + 1, orderIndex + 1, assets, request, hooks, progress),
      );
    }
  }

  const qualification = getQualification(request, assets, rounds, correctness);
  const result: Record<string, unknown> = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    qualification,
    build: {
      sourceCommit: BENCHMARK_BUILD_INFO.gitCommit,
      ortEngine: BENCHMARK_BUILD_INFO.ortEngine,
      ortBundle: BENCHMARK_BUILD_INFO.ortBundle,
      ortWebVersion: ort.env.versions.web,
      ortCommonVersion: ort.env.versions.common,
      liteRtJsVersion: BENCHMARK_BUILD_INFO.litertVersion,
    },
    benchmarkContract: {
      checkpointSha256: OFFICIAL_CHECKPOINT_SHA256,
      semanticBoundary: 'raw detection head; no NMS or end-to-end postprocess',
      ortInputShape: [...ORT_INPUT_SHAPE],
      liteRtInputShape: [...LITERT_INPUT_SHAPE],
      canonicalOutputShape: [...CANONICAL_OUTPUT_SHAPE],
      inputLayoutConversion: 'NCHW to NHWC once before LiteRT.js compilation/timing',
      inputLayoutConversionMilliseconds: assets.inputLayoutConversionMilliseconds,
      inputParsingMilliseconds: assets.inputParsingMilliseconds,
      outputSemanticNormalization:
        'For the immutable TFLite export, multiply normalized box channels [0,4) by 640 after readback, outside timing.',
      precision: 'Explicit LiteRT.js FP32 request plus FP32 ORT model and I/O on one exact shared GPUDevice',
      timedBoundary: request.gpuResidentOutput
        ? 'inference call through GPU queue completion with output left GPU-resident'
        : 'inference call through explicit CPU output readback for every sample',
      wasmArtifactPath: wasmArtifactPath ?? 'default dist path',
      graphCapture: request.graphCapture,
      queueSyncDiagnostic: request.queueSyncDiagnostic,
      wallDecomposition: request.wallDecomposition,
      inputBoundary: request.inputBoundary,
      modelInitializationIncludedInSteadyState: false,
      firstInferenceIncludedInSteadyState: false,
      runtimeSequencing:
        'Arms execute sequentially; each model/session is disposed and GPU queue completion is awaited before the next.',
    },
    configuration: {
      mode: request.mode,
      warmups: request.warmups,
      iterations: request.iterations,
      rounds: request.rounds,
      orderStrategy: request.order,
      stabilizationMilliseconds: request.stabilizationMilliseconds,
      graphCapture: request.graphCapture,
      webGpuTrace: request.webGpuTrace,
      wallDecomposition: request.wallDecomposition,
      gpuResidentOutput: request.gpuResidentOutput,
      randomizedInitialArm: request.order === 'randomized' ? roundOrders[0]?.order[0] : undefined,
      roundOrders,
    },
    artifacts: {
      checkpoint: { sha256: OFFICIAL_CHECKPOINT_SHA256, embeddedInPage: false },
      onnx: summarizeAsset(assets.onnx),
      tflite: summarizeAsset(assets.tflite),
      input: summarizeAsset(assets.input),
    },
    environment: {
      browser,
      adapterProbe,
      runtimeDevice: gpu.diagnostics,
      sameBrowserProcess: true,
      sameGpuDeviceObjectUsedByBothRuntimes: request.mode === 'comparison' && hasCrossRuntimeSharedDeviceProof(rounds),
      liteRtRuntimeInitializationMilliseconds: liteRtInitializationMilliseconds,
    },
    correctness,
    aggregate: {
      ort: aggregateRounds(rounds, 'ort'),
      litert: aggregateRounds(rounds, 'litert'),
    },
    rounds,
    limitations: [
      'LiteRT.js 2.5.3 echoes gpuOptions.precision=fp32 but exposes no post-compilation precision query.',
      'The selected ORT engine is the native C++ WebGPU EP in a repository-built Asyncify WASM bundle.',
      'The caller creates one GPUDevice and passes that exact object to the native ORT WebGPU EP and LiteRT.js.',
      request.graphCapture
        ? 'ORT native WebGPU capture re-encodes saved dispatch records with stable external I/O; it never resubmits a WebGPU command buffer.'
        : 'ORT native WebGPU graph capture is disabled for this run.',
      'The native EP does not expose WebGPU timestamp events through the current JavaScript API; T0-T5 uses API timeline synchronization.',
      'ORT initializes lazily, so its first model compilation can include one-time WASM/runtime initialization.',
      'Cancellation is cooperative between samples and cannot interrupt an in-flight runtime call.',
      'GPU adapter details may be browser-sanitized or unavailable.',
      'Native Intel comparison remains blocked: LiteRT 2.1.6 on Windows selected TITAN V despite the public Intel ' +
        'selector, and direct Intel device injection crashed. No Intel win, tie, or loss is reported.',
    ],
  };
  progress.status('Benchmark complete');
  return result;
};

export const isCancelledError = (error: unknown): boolean => error instanceof CancelledError;

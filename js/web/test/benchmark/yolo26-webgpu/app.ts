// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {
  AssetSelection,
  benchmarkBuildInfo,
  BenchmarkMode,
  BenchmarkRequest,
  InputBoundary,
  isCancelledError,
  runBenchmark,
} from './benchmark-runner';
import {
  EXPECTED_INPUT_SHA256,
  EXPECTED_ONNX_SHA256,
  EXPECTED_TFLITE_SHA256,
  OFFICIAL_CHECKPOINT_SHA256,
  OrderStrategy,
} from './benchmark-utils';

declare global {
  interface Window {
    __benchmarkDone?: boolean;
    __benchmarkError?: string;
    __benchmarkResult?: Record<string, unknown>;
  }
}

const element = <T extends HTMLElement>(id: string): T => {
  const value = document.getElementById(id);
  if (!value) {
    throw new Error(`Missing required page element: #${id}.`);
  }
  return value as T;
};

const onnxFile = element<HTMLInputElement>('onnx-file');
const onnxUrl = element<HTMLInputElement>('onnx-url');
const onnxHash = element<HTMLInputElement>('onnx-hash');
const tfliteFile = element<HTMLInputElement>('tflite-file');
const tfliteUrl = element<HTMLInputElement>('tflite-url');
const tfliteHash = element<HTMLInputElement>('tflite-hash');
const inputFile = element<HTMLInputElement>('input-file');
const inputUrl = element<HTMLInputElement>('input-url');
const inputHash = element<HTMLInputElement>('input-hash');
const warmups = element<HTMLInputElement>('warmups');
const iterations = element<HTMLInputElement>('iterations');
const rounds = element<HTMLInputElement>('rounds');
const order = element<HTMLSelectElement>('order');
const stabilization = element<HTMLInputElement>('stabilization');
const inputBoundary = element<HTMLSelectElement>('input-boundary');
const runOrtButton = element<HTMLButtonElement>('run-ort');
const runLiteRtButton = element<HTMLButtonElement>('run-litert');
const runComparisonButton = element<HTMLButtonElement>('run-comparison');
const cancelButton = element<HTMLButtonElement>('cancel');
const progress = element<HTMLProgressElement>('progress');
const progressText = element<HTMLElement>('progress-text');
const statusText = element<HTMLElement>('status-text');
const errorPanel = element<HTMLElement>('error-panel');
const errorText = element<HTMLElement>('error-text');
const qualification = element<HTMLElement>('qualification');
const correctnessBody = element<HTMLTableSectionElement>('correctness-body');
const aggregateBody = element<HTMLTableSectionElement>('aggregate-body');
const roundsBody = element<HTMLTableSectionElement>('rounds-body');
const environmentOutput = element<HTMLElement>('environment-output');
const jsonOutput = element<HTMLPreElement>('json-output');
const downloadButton = element<HTMLButtonElement>('download-json');
const copyButton = element<HTMLButtonElement>('copy-json');

let cancelRequested = false;
let active = false;
let latestJson = '';

const allRunControls: readonly HTMLButtonElement[] = [runOrtButton, runLiteRtButton, runComparisonButton];

const setActive = (value: boolean): void => {
  active = value;
  for (const button of allRunControls) {
    button.disabled = value;
  }
  cancelButton.disabled = !value;
  document.querySelectorAll<HTMLInputElement | HTMLSelectElement>('[data-benchmark-control]').forEach((control) => {
    control.disabled = value;
  });
};

const clearResults = (): void => {
  errorPanel.hidden = true;
  errorText.textContent = '';
  qualification.textContent = 'Not run';
  qualification.className = 'badge badge-neutral';
  correctnessBody.replaceChildren();
  aggregateBody.replaceChildren();
  roundsBody.replaceChildren();
  environmentOutput.textContent = 'Adapter and runtime diagnostics will appear after a run.';
  jsonOutput.textContent = 'No result yet.';
  downloadButton.disabled = true;
  copyButton.disabled = true;
  latestJson = '';
};

const selectedFile = (input: HTMLInputElement): File | null => input.files?.[0] ?? null;

const assetSelection = (
  fileInput: HTMLInputElement,
  urlInput: HTMLInputElement,
  hashInput: HTMLInputElement,
): AssetSelection => ({
  file: selectedFile(fileInput),
  url: urlInput.value,
  expectedSha256: hashInput.value,
});

const parseInteger = (control: HTMLInputElement, label: string): number => {
  const value = Number(control.value);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`${label} must be an integer.`);
  }
  return value;
};

const createRequest = (mode: BenchmarkMode): BenchmarkRequest => ({
  mode,
  onnx: assetSelection(onnxFile, onnxUrl, onnxHash),
  tflite: assetSelection(tfliteFile, tfliteUrl, tfliteHash),
  input: assetSelection(inputFile, inputUrl, inputHash),
  warmups: parseInteger(warmups, 'Warmups'),
  iterations: parseInteger(iterations, 'Iterations'),
  rounds: parseInteger(rounds, 'Rounds'),
  order: order.value as OrderStrategy,
  stabilizationMilliseconds: parseInteger(stabilization, 'Stabilization interval'),
  inputBoundary: inputBoundary.value as InputBoundary,
  graphCapture: new URLSearchParams(window.location.search).get('graphCapture') === '1',
  queueSyncDiagnostic: new URLSearchParams(window.location.search).get('queueSync') === '1',
  webGpuTrace: new URLSearchParams(window.location.search).get('webGpuTrace') === '1',
  wallDecomposition: new URLSearchParams(window.location.search).get('wallDecomposition') === '1',
  gpuResidentOutput: new URLSearchParams(window.location.search).get('gpuResidentOutput') === '1',
});

const asRecord = (value: unknown): Record<string, unknown> =>
  typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {};

const formatNumber = (value: unknown, digits = 3): string =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';

const addCell = (row: HTMLTableRowElement, text: string): void => {
  const cell = row.insertCell();
  cell.textContent = text;
};

const renderCorrectness = (result: Record<string, unknown>): void => {
  correctnessBody.replaceChildren();
  const correctness = asRecord(result.correctness);
  if (Object.keys(correctness).length === 0) {
    const row = correctnessBody.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 5;
    cell.textContent = 'Cross-runtime parity is only evaluated by Run comparison.';
    return;
  }

  const comparison = asRecord(correctness.comparison);
  const gate = asRecord(correctness.gate);
  const ort = asRecord(correctness.ort);
  const liteRt = asRecord(correctness.litert);
  const row = correctnessBody.insertRow();
  addCell(row, gate.passed === true ? 'PASS' : 'FAIL');
  addCell(row, formatNumber(comparison.maxAbsoluteError, 8));
  addCell(row, formatNumber(comparison.p99AbsoluteError, 8));
  addCell(row, formatNumber(comparison.meanAbsoluteError, 10));
  addCell(
    row,
    `ORT: ${JSON.stringify(ort.sourceShape ?? [])} → ${JSON.stringify(ort.normalizedShape ?? [])} ` +
      `(${String(ort.normalization ?? 'none')}); LiteRT.js: ${JSON.stringify(liteRt.sourceShape ?? [])} → ` +
      `${JSON.stringify(liteRt.normalizedShape ?? [])} (${String(liteRt.normalization ?? 'none')})`,
  );
};

const renderAggregate = (result: Record<string, unknown>): void => {
  aggregateBody.replaceChildren();
  const aggregate = asRecord(result.aggregate);
  for (const [key, label] of [
    ['ort', 'ONNX Runtime WebGPU'],
    ['litert', 'LiteRT.js WebGPU'],
  ] as const) {
    const statistics = asRecord(aggregate[key]);
    if (Object.keys(statistics).length === 0) {
      continue;
    }
    const row = aggregateBody.insertRow();
    addCell(row, label);
    addCell(row, String(statistics.count ?? '—'));
    addCell(row, formatNumber(statistics.p50));
    addCell(row, formatNumber(statistics.p90));
    addCell(row, formatNumber(statistics.mean));
    addCell(row, formatNumber(statistics.standardDeviation));
    addCell(row, formatNumber(statistics.iqr));
    addCell(row, formatNumber(statistics.minimum));
    addCell(row, formatNumber(statistics.maximum));
    addCell(row, formatNumber(statistics.inferencesPerSecond, 2));
  }
};

const renderRounds = (result: Record<string, unknown>): void => {
  roundsBody.replaceChildren();
  if (!Array.isArray(result.rounds)) {
    return;
  }
  for (const rawRound of result.rounds) {
    const round = asRecord(rawRound);
    const statistics = asRecord(round.statistics);
    const row = roundsBody.insertRow();
    addCell(row, String(round.round ?? '—'));
    addCell(row, round.arm === 'ort' ? 'ORT' : 'LiteRT.js');
    addCell(row, String(round.orderInRound ?? '—'));
    addCell(row, formatNumber(round.compileMilliseconds));
    addCell(row, formatNumber(round.firstInferenceMilliseconds));
    addCell(row, formatNumber(statistics.p50));
    addCell(row, formatNumber(statistics.p90));
    addCell(row, formatNumber(statistics.mean));
  }
};

const renderQualification = (result: Record<string, unknown>): void => {
  const qualificationResult = asRecord(result.qualification);
  const status = qualificationResult.status;
  qualification.textContent =
    status === 'presentation-qualified' ? 'Presentation-qualified' : 'Development-only result';
  qualification.className = status === 'presentation-qualified' ? 'badge badge-qualified' : 'badge badge-development';
  if (Array.isArray(qualificationResult.reasons) && qualificationResult.reasons.length > 0) {
    qualification.title = qualificationResult.reasons.map(String).join('\n');
  } else {
    qualification.removeAttribute('title');
  }
};

const renderResult = (result: Record<string, unknown>): void => {
  renderQualification(result);
  renderCorrectness(result);
  renderAggregate(result);
  renderRounds(result);
  environmentOutput.textContent = JSON.stringify(
    {
      build: result.build,
      environment: result.environment,
      artifacts: result.artifacts,
      benchmarkContract: result.benchmarkContract,
      limitations: result.limitations,
    },
    null,
    2,
  );
  latestJson = JSON.stringify(result, null, 2);
  jsonOutput.textContent = latestJson;
  downloadButton.disabled = false;
  copyButton.disabled = false;
};

const showError = (error: unknown): void => {
  const message = error instanceof Error ? `${error.message}\n\n${error.stack ?? ''}` : String(error);
  errorText.textContent = message;
  errorPanel.hidden = false;
};

const startBenchmark = async (mode: BenchmarkMode): Promise<void> => {
  if (active) {
    return;
  }
  clearResults();
  window.__benchmarkDone = false;
  window.__benchmarkError = undefined;
  window.__benchmarkResult = undefined;
  cancelRequested = false;
  setActive(true);
  progress.value = 0;
  progress.max = 1;
  progressText.textContent = 'Starting';
  statusText.textContent = 'Validating configuration';

  try {
    const result = await runBenchmark(createRequest(mode), {
      isCancelled: () => cancelRequested,
      onProgress: (update) => {
        progress.max = Math.max(update.total, 1);
        progress.value = update.completed;
        progressText.textContent = `${update.completed}/${update.total} · ${update.message}`;
      },
      onStatus: (message) => {
        statusText.textContent = message;
      },
    });
    renderResult(result);
    window.__benchmarkResult = result;
    statusText.textContent = 'Complete';
  } catch (error) {
    if (isCancelledError(error)) {
      statusText.textContent = 'Cancelled';
    } else {
      statusText.textContent = 'Failed';
    }
    showError(error);
    window.__benchmarkError = error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error);
  } finally {
    setActive(false);
    window.__benchmarkDone = true;
  }
};

const setDefaults = (): void => {
  onnxHash.value = EXPECTED_ONNX_SHA256;
  tfliteHash.value = EXPECTED_TFLITE_SHA256;
  inputHash.value = EXPECTED_INPUT_SHA256;
  element<HTMLInputElement>('checkpoint-hash').value = OFFICIAL_CHECKPOINT_SHA256;
  element<HTMLElement>('engine-identity').textContent =
    `ORT ${benchmarkBuildInfo.ortEngine} · LiteRT.js ${benchmarkBuildInfo.litertVersion} · source ` +
    benchmarkBuildInfo.gitCommit.slice(0, 12);

  const parameters = new URLSearchParams(window.location.search);
  onnxUrl.value = parameters.get('onnx') ?? './dist/local-assets/yolo26n.onnx';
  tfliteUrl.value = parameters.get('tflite') ?? './dist/local-assets/yolo26n.tflite';
  inputUrl.value = parameters.get('input') ?? './dist/local-assets/input_nchw.npy';
  stabilization.value = parameters.get('stabilization') ?? stabilization.value;
  const requestedOrder = parameters.get('order');
  if (requestedOrder && Array.from(order.options).some((option) => option.value === requestedOrder)) {
    order.value = requestedOrder;
  }
  if (parameters.get('quick') === '1') {
    warmups.value = '2';
    iterations.value = '5';
    rounds.value = '1';
    stabilization.value = '0';
  }
  warmups.value = parameters.get('warmups') ?? warmups.value;
  iterations.value = parameters.get('iterations') ?? iterations.value;
  rounds.value = parameters.get('rounds') ?? rounds.value;
};

runOrtButton.addEventListener('click', () => {
  void startBenchmark('ort');
});
runLiteRtButton.addEventListener('click', () => {
  void startBenchmark('litert');
});
runComparisonButton.addEventListener('click', () => {
  void startBenchmark('comparison');
});
cancelButton.addEventListener('click', () => {
  cancelRequested = true;
  cancelButton.disabled = true;
  statusText.textContent = 'Cancelling after the active runtime call completes';
});
downloadButton.addEventListener('click', () => {
  if (!latestJson) {
    return;
  }
  const url = URL.createObjectURL(new Blob([latestJson], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `yolo26-webgpu-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  link.click();
  URL.revokeObjectURL(url);
});
copyButton.addEventListener('click', () => {
  if (latestJson) {
    void navigator.clipboard.writeText(latestJson);
  }
});

setDefaults();
clearResults();
setActive(false);
if (new URLSearchParams(window.location.search).get('auto') === '1') {
  void startBenchmark('comparison');
}

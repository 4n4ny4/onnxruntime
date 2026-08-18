// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export const OFFICIAL_CHECKPOINT_SHA256 = '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef';
export const EXPECTED_ONNX_SHA256 = '5b551efe169f8656e883a677740bd92e2eb73bdd38ea0b02d3f596c9a9d3e807';
export const EXPECTED_TFLITE_SHA256 = 'aec199b2383caf5fe356afbf966d63364bc1b66e0620b599a177501fd99662cf';
export const EXPECTED_INPUT_SHA256 = 'a95ffe2ea506d816578be4da69c1fe76b9ef2a658ab79e3fc101c3fdbb5cb60b';

export const ORT_INPUT_SHAPE = [1, 3, 640, 640] as const;
export const LITERT_INPUT_SHAPE = [1, 640, 640, 3] as const;
export const CANONICAL_OUTPUT_SHAPE = [1, 84, 8400] as const;
export const TRANSPOSED_OUTPUT_SHAPE = [1, 8400, 84] as const;

export const QUALIFIED_MAX_ABSOLUTE_ERROR = 1e-3;
export const QUALIFIED_P99_ABSOLUTE_ERROR = 1e-4;

export interface Statistics {
  readonly count: number;
  readonly p50: number;
  readonly p90: number;
  readonly mean: number;
  readonly standardDeviation: number;
  readonly iqr: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly inferencesPerSecond: number;
}

export interface ParsedNpy {
  readonly data: Float32Array;
  readonly shape: readonly number[];
  readonly dtype: 'float32';
}

export interface NormalizedOutput {
  readonly data: Float32Array;
  readonly sourceShape: readonly number[];
  readonly normalizedShape: readonly number[];
  readonly normalization: 'none' | 'transpose-[1,8400,84]-to-[1,84,8400]';
}

export interface OutputComparison {
  readonly elementCount: number;
  readonly maxAbsoluteError: number;
  readonly p99AbsoluteError: number;
  readonly meanAbsoluteError: number;
  readonly nonFiniteCount: number;
}

export type ArmName = 'ort' | 'litert';
export type OrderStrategy = 'alternating' | 'randomized' | 'ort-first' | 'litert-first';

const shapesEqual = (actual: readonly number[], expected: readonly number[]): boolean =>
  actual.length === expected.length && actual.every((value, index) => value === expected[index]);

const product = (values: readonly number[]): number => values.reduce((total, value) => total * value, 1);

const quantileSorted = (sortedValues: ArrayLike<number>, quantile: number): number => {
  if (sortedValues.length === 0) {
    throw new Error('Cannot calculate a quantile for an empty sample set.');
  }

  const position = (sortedValues.length - 1) * quantile;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  if (lowerIndex === upperIndex) {
    return sortedValues[lowerIndex];
  }

  const weight = position - lowerIndex;
  return sortedValues[lowerIndex] * (1 - weight) + sortedValues[upperIndex] * weight;
};

export const calculateStatistics = (samples: readonly number[]): Statistics => {
  if (samples.length === 0) {
    throw new Error('At least one latency sample is required.');
  }
  if (samples.some((sample) => !Number.isFinite(sample) || sample < 0)) {
    throw new Error('Latency samples must be finite, non-negative numbers.');
  }

  const sorted = [...samples].sort((left, right) => left - right);
  const mean = samples.reduce((sum, sample) => sum + sample, 0) / samples.length;
  const variance =
    samples.length > 1 ? samples.reduce((sum, sample) => sum + (sample - mean) ** 2, 0) / (samples.length - 1) : 0;
  const q1 = quantileSorted(sorted, 0.25);
  const q3 = quantileSorted(sorted, 0.75);

  return {
    count: samples.length,
    p50: quantileSorted(sorted, 0.5),
    p90: quantileSorted(sorted, 0.9),
    mean,
    standardDeviation: Math.sqrt(variance),
    iqr: q3 - q1,
    minimum: sorted[0],
    maximum: sorted[sorted.length - 1],
    inferencesPerSecond: mean === 0 ? Number.POSITIVE_INFINITY : 1000 / mean,
  };
};

const parseNpyShape = (header: string): number[] => {
  const shapeMatch = /['"]shape['"]\s*:\s*\(([^)]*)\)/.exec(header);
  if (!shapeMatch) {
    throw new Error('The NPY header does not contain a shape tuple.');
  }

  const dimensions = shapeMatch[1]
    .split(',')
    .map((dimension) => dimension.trim())
    .filter((dimension) => dimension.length > 0)
    .map((dimension) => Number.parseInt(dimension, 10));
  if (dimensions.length === 0 || dimensions.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0)) {
    throw new Error(`The NPY shape is invalid: (${shapeMatch[1]}).`);
  }
  return dimensions;
};

export const parseFloat32Npy = (bytes: Uint8Array): ParsedNpy => {
  if (
    bytes.length < 10 ||
    bytes[0] !== 0x93 ||
    bytes[1] !== 0x4e ||
    bytes[2] !== 0x55 ||
    bytes[3] !== 0x4d ||
    bytes[4] !== 0x50 ||
    bytes[5] !== 0x59
  ) {
    throw new Error('Input is not a NumPy NPY file.');
  }

  const majorVersion = bytes[6];
  const headerLengthOffset = 8;
  let headerOffset: number;
  let headerLength: number;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (majorVersion === 1) {
    headerOffset = 10;
    headerLength = view.getUint16(headerLengthOffset, true);
  } else if (majorVersion === 2 || majorVersion === 3) {
    if (bytes.length < 12) {
      throw new Error('The NPY file is truncated before its header length.');
    }
    headerOffset = 12;
    headerLength = view.getUint32(headerLengthOffset, true);
  } else {
    throw new Error(`Unsupported NPY version ${majorVersion}.${bytes[7]}.`);
  }

  const dataOffset = headerOffset + headerLength;
  if (dataOffset > bytes.length) {
    throw new Error('The NPY file is truncated in its header.');
  }

  const header = new TextDecoder(majorVersion === 3 ? 'utf-8' : 'latin1').decode(
    bytes.subarray(headerOffset, dataOffset),
  );
  const dtypeMatch = /['"]descr['"]\s*:\s*['"]([^'"]+)['"]/.exec(header);
  if (!dtypeMatch || (dtypeMatch[1] !== '<f4' && dtypeMatch[1] !== '=f4')) {
    throw new Error(`Expected a little-endian float32 NPY file, got ${dtypeMatch?.[1] ?? 'unknown dtype'}.`);
  }
  if (/['"]fortran_order['"]\s*:\s*True/.test(header)) {
    throw new Error('Fortran-order NPY arrays are not supported.');
  }

  const shape = parseNpyShape(header);
  const elementCount = product(shape);
  const expectedByteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
  if (bytes.length - dataOffset !== expectedByteLength) {
    throw new Error(
      `NPY payload size mismatch: expected ${expectedByteLength} bytes for [${shape.join(',')}], ` +
        `got ${bytes.length - dataOffset}.`,
    );
  }

  const copiedData = bytes.buffer.slice(bytes.byteOffset + dataOffset, bytes.byteOffset + bytes.length);
  return { data: new Float32Array(copiedData), shape, dtype: 'float32' };
};

export const requireShape = (actual: readonly number[], expected: readonly number[], label: string): void => {
  if (!shapesEqual(actual, expected)) {
    throw new Error(`${label} must have shape [${expected.join(',')}], got [${actual.join(',')}].`);
  }
};

export const transposeNchwToNhwc = (input: Float32Array, shape: readonly number[] = ORT_INPUT_SHAPE): Float32Array => {
  requireShape(shape, ORT_INPUT_SHAPE, 'Input tensor');
  if (input.length !== product(shape)) {
    throw new Error(`Input tensor has ${input.length} values; expected ${product(shape)}.`);
  }

  const [, channels, height, width] = shape;
  const output = new Float32Array(input.length);
  for (let channel = 0; channel < channels; channel++) {
    const channelOffset = channel * height * width;
    for (let row = 0; row < height; row++) {
      const rowOffset = row * width;
      for (let column = 0; column < width; column++) {
        output[(rowOffset + column) * channels + channel] = input[channelOffset + rowOffset + column];
      }
    }
  }
  return output;
};

export const normalizeOutput = (data: Float32Array, shape: readonly number[]): NormalizedOutput => {
  if (shapesEqual(shape, CANONICAL_OUTPUT_SHAPE)) {
    if (data.length !== product(shape)) {
      throw new Error(`Output has ${data.length} values; expected ${product(shape)}.`);
    }
    return {
      data,
      sourceShape: [...shape],
      normalizedShape: [...CANONICAL_OUTPUT_SHAPE],
      normalization: 'none',
    };
  }

  if (!shapesEqual(shape, TRANSPOSED_OUTPUT_SHAPE)) {
    throw new Error(
      `Output must have shape [${CANONICAL_OUTPUT_SHAPE.join(',')}] or ` +
        `[${TRANSPOSED_OUTPUT_SHAPE.join(',')}], got [${shape.join(',')}].`,
    );
  }
  if (data.length !== product(shape)) {
    throw new Error(`Output has ${data.length} values; expected ${product(shape)}.`);
  }

  const output = new Float32Array(data.length);
  const rowCount = TRANSPOSED_OUTPUT_SHAPE[1];
  const columnCount = TRANSPOSED_OUTPUT_SHAPE[2];
  for (let row = 0; row < rowCount; row++) {
    for (let column = 0; column < columnCount; column++) {
      output[column * rowCount + row] = data[row * columnCount + column];
    }
  }
  return {
    data: output,
    sourceShape: [...shape],
    normalizedShape: [...CANONICAL_OUTPUT_SHAPE],
    normalization: 'transpose-[1,8400,84]-to-[1,84,8400]',
  };
};

export const scaleYoloBoxChannels = (
  data: Float32Array,
  shape: readonly number[] = CANONICAL_OUTPUT_SHAPE,
  scale = 640,
): Float32Array => {
  requireShape(shape, CANONICAL_OUTPUT_SHAPE, 'YOLO raw-head output');
  if (data.length !== product(shape)) {
    throw new Error(`Output has ${data.length} values; expected ${product(shape)}.`);
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`YOLO box scale must be a positive finite number, got ${scale}.`);
  }

  const output = new Float32Array(data);
  const candidateCount = CANONICAL_OUTPUT_SHAPE[2];
  for (let channel = 0; channel < 4; channel++) {
    const channelOffset = channel * candidateCount;
    for (let candidate = 0; candidate < candidateCount; candidate++) {
      output[channelOffset + candidate] *= scale;
    }
  }
  return output;
};

export const compareOutputs = (left: Float32Array, right: Float32Array): OutputComparison => {
  if (left.length !== right.length) {
    throw new Error(`Output element counts differ: ${left.length} versus ${right.length}.`);
  }
  if (left.length === 0) {
    throw new Error('Outputs are empty.');
  }

  const errors = new Float32Array(left.length);
  let sum = 0;
  let maximum = 0;
  let nonFiniteCount = 0;
  for (let index = 0; index < left.length; index++) {
    const error = Math.abs(left[index] - right[index]);
    if (!Number.isFinite(error)) {
      nonFiniteCount++;
      errors[index] = Number.POSITIVE_INFINITY;
      maximum = Number.POSITIVE_INFINITY;
      continue;
    }
    errors[index] = error;
    sum += error;
    maximum = Math.max(maximum, error);
  }
  errors.sort();

  return {
    elementCount: left.length,
    maxAbsoluteError: maximum,
    p99AbsoluteError: quantileSorted(errors, 0.99),
    meanAbsoluteError: nonFiniteCount === 0 ? sum / left.length : Number.POSITIVE_INFINITY,
    nonFiniteCount,
  };
};

export const normalizeSha256 = (value: string): string => {
  const normalized = value.trim().toLowerCase();
  if (normalized.length > 0 && !/^[0-9a-f]{64}$/.test(normalized)) {
    throw new Error(`SHA256 values must contain exactly 64 hexadecimal characters: "${value}".`);
  }
  return normalized;
};

export const sha256Hex = async (bytes: Uint8Array): Promise<string> => {
  const digestInput = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(digestInput).set(bytes);
  const digest = await crypto.subtle.digest('SHA-256', digestInput);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join('');
};

export const validateSha256 = (actual: string, expected: string, label: string): void => {
  const normalizedExpected = normalizeSha256(expected);
  if (normalizedExpected && actual !== normalizedExpected) {
    throw new Error(`${label} SHA256 mismatch: expected ${normalizedExpected}, got ${actual}.`);
  }
};

export const getRoundOrder = (
  roundIndex: number,
  strategy: OrderStrategy,
  randomValue: number = Math.random(),
): readonly ArmName[] => {
  switch (strategy) {
    case 'alternating':
      return roundIndex % 2 === 0 ? ['ort', 'litert'] : ['litert', 'ort'];
    case 'randomized':
      return (roundIndex + (randomValue < 0.5 ? 0 : 1)) % 2 === 0 ? ['ort', 'litert'] : ['litert', 'ort'];
    case 'ort-first':
      return ['ort', 'litert'];
    case 'litert-first':
      return ['litert', 'ort'];
    default:
      throw new Error('Unsupported runtime order strategy.');
  }
};

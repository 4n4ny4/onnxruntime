// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';

import {
  calculateStatistics,
  compareOutputs,
  getRoundOrder,
  normalizeOutput,
  normalizeSha256,
  parseFloat32Npy,
  scaleYoloBoxChannels,
  transposeNchwToNhwc,
  validateSha256,
} from '../benchmark/yolo26-webgpu/benchmark-utils';

const createNpy = (
  values: readonly number[],
  shape: readonly number[],
  options: { readonly dtype?: string; readonly fortranOrder?: boolean } = {},
): Uint8Array => {
  const dtype = options.dtype ?? '<f4';
  const fortranOrder = options.fortranOrder ?? false;
  const shapeText = shape.length === 1 ? `${shape[0]},` : shape.join(', ');
  const headerWithoutPadding =
    `{'descr': '${dtype}', 'fortran_order': ${fortranOrder ? 'True' : 'False'}, ` + `'shape': (${shapeText}), }`;
  const preambleLength = 10;
  const paddingLength = (16 - ((preambleLength + headerWithoutPadding.length + 1) % 16)) % 16;
  const header = `${headerWithoutPadding}${' '.repeat(paddingLength)}\n`;
  const result = new Uint8Array(preambleLength + header.length + values.length * Float32Array.BYTES_PER_ELEMENT);
  result.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(result.buffer).setUint16(8, header.length, true);
  result.set(new TextEncoder().encode(header), preambleLength);
  new Float32Array(result.buffer, preambleLength + header.length, values.length).set(values);
  return result;
};

describe('YOLO26 WebGPU benchmark utilities', () => {
  it('calculates latency statistics with interpolated percentiles and sample standard deviation', () => {
    const statistics = calculateStatistics([1, 2, 3, 4, 5]);
    expect(statistics.count).to.equal(5);
    expect(statistics.p50).to.equal(3);
    expect(statistics.p90).to.equal(4.6);
    expect(statistics.mean).to.equal(3);
    expect(statistics.standardDeviation).to.equal(Math.sqrt(2.5));
    expect(statistics.iqr).to.equal(2);
    expect(statistics.minimum).to.equal(1);
    expect(statistics.maximum).to.equal(5);
    expect(statistics.inferencesPerSecond).to.equal(1000 / 3);
  });

  it('rejects invalid latency samples', () => {
    expect(() => calculateStatistics([])).to.throw('At least one latency sample');
    expect(() => calculateStatistics([1, -1])).to.throw('finite, non-negative');
  });

  it('parses a C-order little-endian float32 NPY file', () => {
    const parsed = parseFloat32Npy(createNpy([1, 2, 3, 4], [1, 2, 2]));
    expect(parsed.dtype).to.equal('float32');
    expect(parsed.shape).to.deep.equal([1, 2, 2]);
    expect([...parsed.data]).to.deep.equal([1, 2, 3, 4]);
  });

  it('rejects unsupported NPY dtypes, layouts, and payload sizes', () => {
    expect(() => parseFloat32Npy(createNpy([1], [1], { dtype: '>f4' }))).to.throw('little-endian float32');
    expect(() => parseFloat32Npy(createNpy([1], [1], { fortranOrder: true }))).to.throw('Fortran-order');
    expect(() => parseFloat32Npy(createNpy([1], [2]))).to.throw('payload size mismatch');
  });

  it('transposes NCHW input to NHWC once outside inference', () => {
    const input = new Float32Array(1 * 3 * 640 * 640);
    input[0] = 1;
    input[640 * 640] = 2;
    input[2 * 640 * 640] = 3;
    const output = transposeNchwToNhwc(input);
    expect([...output.subarray(0, 3)]).to.deep.equal([1, 2, 3]);
  });

  it('normalizes a transposed raw head to [1,84,8400]', () => {
    const transposed = new Float32Array(1 * 8400 * 84);
    transposed[0] = 1;
    transposed[84] = 2;
    transposed[1] = 3;
    const normalized = normalizeOutput(transposed, [1, 8400, 84]);
    expect(normalized.normalization).to.equal('transpose-[1,8400,84]-to-[1,84,8400]');
    expect(normalized.data[0]).to.equal(1);
    expect(normalized.data[1]).to.equal(2);
    expect(normalized.data[8400]).to.equal(3);
  });

  it('scales only the four normalized YOLO box channels', () => {
    const output = new Float32Array(1 * 84 * 8400);
    output[0] = 0.5;
    output[3 * 8400 + 17] = 0.25;
    output[4 * 8400] = 0.75;
    const scaled = scaleYoloBoxChannels(output);
    expect(scaled[0]).to.equal(320);
    expect(scaled[3 * 8400 + 17]).to.equal(160);
    expect(scaled[4 * 8400]).to.equal(0.75);
    expect(output[0]).to.equal(0.5);
  });

  it('reports output error metrics', () => {
    const comparison = compareOutputs(new Float32Array([0, 1, 2, 3]), new Float32Array([0, 2, 4, 6]));
    expect(comparison.maxAbsoluteError).to.equal(3);
    expect(comparison.meanAbsoluteError).to.equal(1.5);
    expect(comparison.p99AbsoluteError).to.be.closeTo(2.97, 1e-6);
    expect(comparison.nonFiniteCount).to.equal(0);
  });

  it('validates SHA256 values', () => {
    const hash = 'A'.repeat(64);
    expect(normalizeSha256(` ${hash} `)).to.equal('a'.repeat(64));
    expect(() => normalizeSha256('abcd')).to.throw('64 hexadecimal');
    expect(() => validateSha256('a'.repeat(64), 'b'.repeat(64), 'model')).to.throw('model SHA256 mismatch');
  });

  it('alternates and randomizes runtime order', () => {
    expect(getRoundOrder(0, 'alternating')).to.deep.equal(['ort', 'litert']);
    expect(getRoundOrder(1, 'alternating')).to.deep.equal(['litert', 'ort']);
    expect(getRoundOrder(0, 'randomized', 0.1)).to.deep.equal(['ort', 'litert']);
    expect(getRoundOrder(1, 'randomized', 0.1)).to.deep.equal(['litert', 'ort']);
    expect(getRoundOrder(0, 'randomized', 0.9)).to.deep.equal(['litert', 'ort']);
    expect(getRoundOrder(1, 'randomized', 0.9)).to.deep.equal(['ort', 'litert']);
  });
});

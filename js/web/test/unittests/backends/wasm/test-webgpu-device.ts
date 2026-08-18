// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';
import { env, InferenceSession, Tensor } from 'onnxruntime-common';

const MUL_MODEL = Uint8Array.from([
  8, 3, 18, 6, 99, 104, 101, 110, 116, 97, 58, 112, 10, 21, 10, 1, 88, 10, 1, 87, 18, 1, 89, 26, 5, 109, 117,
  108, 95, 49, 34, 3, 77, 117, 108, 18, 8, 109, 117, 108, 32, 116, 101, 115, 116, 42, 35, 8, 3, 8, 2, 16, 1,
  34, 24, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64, 66, 1,
  87, 90, 19, 10, 1, 88, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 3, 10, 2, 8, 2, 98, 19, 10, 1, 89, 18,
  14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 3, 10, 2, 8, 2, 66, 4, 10, 0, 16, 7,
]);
const INPUT_DATA = new Float32Array([1, 2, 3, 4, 5, 6]);
const EXPECTED_OUTPUT = [1, 4, 9, 16, 25, 36];

const createSession = (device?: GPUDevice): Promise<InferenceSession> =>
  InferenceSession.create(MUL_MODEL, {
    executionProviders: [{ name: 'webgpu', device }],
    extra: { session: { disable_cpu_ep_fallback: '1' } },
  });

const runSession = async (session: InferenceSession, device?: GPUDevice): Promise<void> => {
  let input: Tensor;
  if (device) {
    const gpuBuffer = device.createBuffer({
      size: INPUT_DATA.byteLength,
      // eslint-disable-next-line no-bitwise
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(gpuBuffer, 0, INPUT_DATA);
    input = Tensor.fromGpuBuffer(gpuBuffer, {
      dataType: 'float32',
      dims: [3, 2],
      dispose: () => gpuBuffer.destroy(),
    });
  } else {
    input = new Tensor('float32', INPUT_DATA, [3, 2]);
  }

  try {
    const outputs = await session.run({ X: input });
    const output = outputs.Y;
    try {
      const data = await output.getData();
      expect(data).to.be.instanceOf(Float32Array);
      if (!(data instanceof Float32Array)) {
        throw new Error(`Expected float32 output, got ${output.type}.`);
      }
      expect(Array.from(data)).to.deep.equal(EXPECTED_OUTPUT);
    } finally {
      output.dispose();
    }
  } finally {
    input.dispose();
  }
};

const describeWebGpu = typeof navigator !== 'undefined' && navigator.gpu ? describe : describe.skip;

describeWebGpu('#UnitTest# - wasm - WebGPU EP device registration', () => {
  let device: GPUDevice;

  before(async () => {
    env.wasm.numThreads = 1;
    env.wasm.proxy = false;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    expect(adapter, 'a WebGPU adapter is required').to.not.equal(null);
    device = await adapter!.requestDevice();
  });

  after(() => {
    device.destroy();
  });

  it('creates and runs a session with the default ORT device', async () => {
    const session = await createSession();
    try {
      await runSession(session);
      const defaultDevice = await env.webgpu.device;
      expect(defaultDevice).to.have.property('createBuffer').that.is.a('function');
    } finally {
      await session.release();
    }
  });

  it('uses timed waits with a caller-provided device and GPU input', async () => {
    const session = await createSession(device);
    try {
      await runSession(session, device);
    } finally {
      await session.release();
    }
  });

  it('repeatedly creates and releases sessions on the same caller device', async () => {
    for (let iteration = 0; iteration < 3; iteration++) {
      const session = await createSession(device);
      try {
        await runSession(session, device);
      } finally {
        await session.release();
      }
    }
  });

  it('clears caller-device registration after session creation failure', async () => {
    let creationError: unknown;
    try {
      await InferenceSession.create(Uint8Array.of(0), {
        executionProviders: [{ name: 'webgpu', device }],
      });
    } catch (error) {
      creationError = error;
    }
    expect(creationError).to.be.instanceOf(Error);

    const session = await createSession(device);
    try {
      await runSession(session, device);
    } finally {
      await session.release();
    }
  });
});

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

type TraceEvent =
  | {
      readonly type: 'buffer';
      readonly phase: string;
      readonly id: string;
      readonly size: number;
      readonly usage: number;
      readonly usageNames: readonly string[];
      readonly label?: string;
    }
  | {
      readonly type: 'shader';
      readonly phase: string;
      readonly id: string;
      readonly label?: string;
    }
  | {
      readonly type: 'pipeline';
      readonly phase: string;
      readonly id: string;
      readonly shaderId?: string;
      readonly entryPoint?: string;
      readonly label?: string;
      readonly layout: 'auto' | 'explicit';
      readonly constantCount: number;
      readonly constants: Readonly<Record<string, number>>;
    }
  | {
      readonly type: 'bindGroup';
      readonly phase: string;
      readonly id: string;
      readonly label?: string;
      readonly entries: ReadonlyArray<{
        readonly binding: number;
        readonly bufferId?: string;
        readonly offset?: number;
        readonly size?: number;
      }>;
    }
  | {
      readonly type: 'dispatch';
      readonly phase: string;
      readonly passId: string;
      readonly pipelineId?: string;
      readonly x?: number;
      readonly y?: number;
      readonly z?: number;
      readonly indirectBufferId?: string;
      readonly indirectOffset?: number;
      readonly bindGroupIds?: readonly string[];
    }
  | {
      readonly type: 'copyBufferToBuffer';
      readonly phase: string;
      readonly sourceId?: string;
      readonly destinationId?: string;
      readonly size: number;
    }
  | {
      readonly type: 'submit';
      readonly phase: string;
      readonly timestamp: number;
      readonly commandBufferCount: number;
    }
  | {
      readonly type: 'mapAsyncResolved';
      readonly phase: string;
      readonly timestamp: number;
      readonly bufferId?: string;
    };

export interface WebGpuTraceMarker {
  readonly eventIndex: number;
  readonly counters: Readonly<Record<string, number>>;
}

interface ShaderCapture {
  readonly id: string;
  readonly label?: string;
  code?: string;
}

interface PipelineCapture {
  readonly id: string;
  readonly shaderId?: string;
  readonly entryPoint?: string;
  readonly label?: string;
  readonly layout: 'auto' | 'explicit';
  readonly constantCount: number;
  readonly constants: Readonly<Record<string, number>>;
}

interface BufferCapture {
  readonly id: string;
  readonly size: number;
  readonly usage: number;
  readonly usageNames: readonly string[];
  readonly label?: string;
}

interface BindGroupCapture {
  readonly id: string;
  readonly label?: string;
  readonly entries: ReadonlyArray<{
    readonly binding: number;
    readonly bufferId?: string;
    readonly offset?: number;
    readonly size?: number;
  }>;
}

interface ComputePassState {
  readonly id: string;
  readonly bindGroupIds: Map<number, string>;
  pipelineId?: string;
}

const now = (): number => performance.now();

const countMatches = (value: string, expression: RegExp): number => [...value.matchAll(expression)].length;

const splitTopLevelComma = (value: string): [string, string] | undefined => {
  let depth = 0;
  for (let index = 0; index < value.length; index++) {
    const character = value[index];
    if (character === '<' || character === '(') {
      depth++;
    } else if (character === '>' || character === ')') {
      depth--;
    } else if (character === ',' && depth === 0) {
      return [value.slice(0, index), value.slice(index + 1)];
    }
  }
  return undefined;
};

const estimateWgslTypeBytes = (rawType: string, constants: ReadonlyMap<string, number>): number | undefined => {
  let type = rawType.replace(/\s+/g, '');
  for (const [name, value] of constants) {
    type = type.replace(new RegExp(`\\b${name}\\b`, 'g'), String(value));
  }

  const scalarBytes = new Map([
    ['f16', 2],
    ['f32', 4],
    ['i32', 4],
    ['u32', 4],
  ]);
  const scalar = scalarBytes.get(type);
  if (scalar) {
    return scalar;
  }

  const vector = /^vec([234])<(.+)>$/.exec(type);
  if (vector) {
    const elementBytes = estimateWgslTypeBytes(vector[2], constants);
    if (!elementBytes) {
      return undefined;
    }
    const components = Number(vector[1]);
    return components === 3 ? elementBytes * 4 : elementBytes * components;
  }

  const matrix = /^mat([234])x([234])<(.+)>$/.exec(type);
  if (matrix) {
    const columns = Number(matrix[1]);
    const rows = Number(matrix[2]);
    const elementBytes = estimateWgslTypeBytes(matrix[3], constants);
    if (!elementBytes) {
      return undefined;
    }
    const columnBytes = rows === 3 ? elementBytes * 4 : elementBytes * rows;
    return columns * columnBytes;
  }

  if (type.startsWith('array<') && type.endsWith('>')) {
    const parts = splitTopLevelComma(type.slice(6, -1));
    if (!parts) {
      return undefined;
    }
    const elementBytes = estimateWgslTypeBytes(parts[0], constants);
    const count = Number(parts[1].replace(/u$/, ''));
    return elementBytes && Number.isSafeInteger(count) ? elementBytes * count : undefined;
  }

  return undefined;
};

const sha256Hex = async (value: string): Promise<string> => {
  const bytes = new TextEncoder().encode(value);
  const input = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(input).set(bytes);
  const digest = await crypto.subtle.digest('SHA-256', input);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
};

const shaderMetadata = async (capture: ShaderCapture): Promise<Record<string, unknown>> => {
  const code = capture.code ?? '';
  const constants = new Map<string, number>();
  for (const match of code.matchAll(/\bconst\s+([A-Za-z_]\w*)[^=]*=\s*(\d+)u?\s*;/g)) {
    constants.set(match[1], Number(match[2]));
  }

  const workgroupDeclarations = [...code.matchAll(/var<workgroup>\s+\w+\s*:\s*([^;]+);/g)].map((match) => match[1]);
  const workgroupDeclarationBytes = workgroupDeclarations.map((type) => estimateWgslTypeBytes(type, constants));
  const knownWorkgroupBytes = workgroupDeclarationBytes.filter((value): value is number => value !== undefined);
  const workgroupSizes = [...code.matchAll(/@workgroup_size\s*\(([^)]*)\)/g)].map((match) =>
    match[1].split(',').map((value) => value.trim()),
  );
  const entryPoints = [...code.matchAll(/@compute[\s\S]{0,160}?fn\s+([A-Za-z_]\w*)\s*\(/g)].map((match) => match[1]);

  const result: Record<string, unknown> = {
    id: capture.id,
    label: capture.label,
    sha256: await sha256Hex(code),
    codeLength: code.length,
    entryPoints,
    workgroupSizes,
    bindingCount: countMatches(code, /@group\s*\(\s*\d+\s*\)\s*@binding\s*\(\s*\d+\s*\)/g),
    storageBindingCount: countMatches(code, /var<storage\b/g),
    uniformBindingCount: countMatches(code, /var<uniform\b/g),
    workgroupDeclarationCount: workgroupDeclarations.length,
    workgroupStorageBytes:
      knownWorkgroupBytes.length === workgroupDeclarations.length
        ? knownWorkgroupBytes.reduce((sum, value) => sum + value, 0)
        : undefined,
    workgroupStorageUnknownDeclarations: workgroupDeclarations.length - knownWorkgroupBytes.length,
    usesVec4: /\bvec4\s*</.test(code),
    usesMat4: /\bmat4x4\s*</.test(code),
    usesSubgroups: /\bsubgroup\w*/.test(code),
    expCallCount: countMatches(code, /\bexp\s*\(/g),
    exp2CallCount: countMatches(code, /\bexp2\s*\(/g),
    log2CallCount: countMatches(code, /\blog2\s*\(/g),
    isContiguousVec4Split: code.includes('output_0[') && code.includes('=input[global_idx]'),
    isContiguousVec4Concat: code.includes('output[global_idx]=input_'),
    isScalarConcat: code.includes('calculate_input_index') && code.includes('assign_output_data'),
    isScalarSplit: code.includes('calculate_output_index') && code.includes('write_buffer_data'),
  };
  capture.code = undefined;
  return result;
};

const bufferUsageNames = (usage: number): string[] => {
  const flags: ReadonlyArray<readonly [number, string]> = [
    [GPUBufferUsage.MAP_READ, 'MAP_READ'],
    [GPUBufferUsage.MAP_WRITE, 'MAP_WRITE'],
    [GPUBufferUsage.COPY_SRC, 'COPY_SRC'],
    [GPUBufferUsage.COPY_DST, 'COPY_DST'],
    [GPUBufferUsage.INDEX, 'INDEX'],
    [GPUBufferUsage.VERTEX, 'VERTEX'],
    [GPUBufferUsage.UNIFORM, 'UNIFORM'],
    [GPUBufferUsage.STORAGE, 'STORAGE'],
    [GPUBufferUsage.INDIRECT, 'INDIRECT'],
    [GPUBufferUsage.QUERY_RESOLVE, 'QUERY_RESOLVE'],
  ];
  // eslint-disable-next-line no-bitwise
  return flags.filter(([flag]) => (usage & flag) !== 0).map(([, name]) => name);
};

export class WebGpuApiTrace {
  private readonly counters: Record<string, number> = {};
  private readonly events: TraceEvent[] = [];
  private readonly shaderCaptures = new Map<string, ShaderCapture>();
  private readonly pipelineCaptures = new Map<string, PipelineCapture>();
  private readonly bufferCaptures = new Map<string, BufferCapture>();
  private readonly bindGroupCaptures = new Map<string, BindGroupCapture>();
  private readonly shaderIds = new WeakMap<GPUShaderModule, string>();
  private readonly pipelineIds = new WeakMap<GPUComputePipeline, string>();
  private readonly bufferIds = new WeakMap<GPUBuffer, string>();
  private readonly bindGroupIds = new WeakMap<GPUBindGroup, string>();
  private readonly encoderIds = new WeakMap<GPUCommandEncoder, string>();
  private readonly passStates = new WeakMap<GPUComputePassEncoder, ComputePassState>();
  private readonly restores: Array<() => void> = [];
  private readonly patchedMembers = new WeakMap<object, Set<PropertyKey>>();
  private phase = 'compile';
  private nextShaderId = 1;
  private nextPipelineId = 1;
  private nextBufferId = 1;
  private nextBindGroupId = 1;
  private nextEncoderId = 1;
  private nextPassId = 1;
  private installed = false;

  constructor(private readonly device: GPUDevice) {}

  private increment(name: string, value = 1): void {
    this.counters[name] = (this.counters[name] ?? 0) + value;
  }

  private timed<T>(counterName: string, operation: () => T): T {
    const start = now();
    try {
      return operation();
    } finally {
      this.increment(counterName, now() - start);
    }
  }

  private patchPrototype<T extends object, K extends keyof T>(
    sample: T,
    key: K,
    replacementFactory: (original: T[K]) => T[K],
  ): void {
    let owner: object | null = sample;
    while (owner && !Object.prototype.hasOwnProperty.call(owner, key)) {
      owner = Object.getPrototypeOf(owner) as object | null;
    }
    if (!owner) {
      throw new Error(`Cannot trace WebGPU method ${String(key)}: no property descriptor was found.`);
    }

    const patched = this.patchedMembers.get(owner) ?? new Set<PropertyKey>();
    if (patched.has(key)) {
      return;
    }
    const descriptor = Object.getOwnPropertyDescriptor(owner, key);
    if (!descriptor || typeof descriptor.value !== 'function' || !descriptor.configurable) {
      throw new Error(`Cannot trace WebGPU method ${String(key)}: descriptor is not patchable.`);
    }

    const replacement = replacementFactory(descriptor.value as T[K]);
    Object.defineProperty(owner, key, { ...descriptor, value: replacement });
    patched.add(key);
    this.patchedMembers.set(owner, patched);
    this.restores.push(() => {
      Object.defineProperty(owner as object, key, descriptor);
      patched.delete(key);
    });
  }

  private captureBuffer(buffer: GPUBuffer, descriptor: GPUBufferDescriptor): void {
    const id = `buffer-${this.nextBufferId++}`;
    const capture: BufferCapture = {
      id,
      size: descriptor.size,
      usage: descriptor.usage,
      usageNames: bufferUsageNames(descriptor.usage),
      label: descriptor.label,
    };
    this.bufferIds.set(buffer, id);
    this.bufferCaptures.set(id, capture);
    this.events.push({
      type: 'buffer',
      phase: this.phase,
      ...capture,
    });
    this.patchBufferPrototype(buffer);
  }

  private captureShader(module: GPUShaderModule, descriptor: GPUShaderModuleDescriptor): void {
    const id = `shader-${this.nextShaderId++}`;
    this.shaderIds.set(module, id);
    this.shaderCaptures.set(id, { id, label: descriptor.label, code: descriptor.code });
    this.events.push({ type: 'shader', phase: this.phase, id, label: descriptor.label });
  }

  private capturePipeline(pipeline: GPUComputePipeline, descriptor: GPUComputePipelineDescriptor): void {
    const id = `pipeline-${this.nextPipelineId++}`;
    const shaderId = this.shaderIds.get(descriptor.compute.module);
    const constants = Object.fromEntries(
      Object.entries(descriptor.compute.constants ?? {}).map(([name, value]) => [name, Number(value)]),
    );
    const capture: PipelineCapture = {
      id,
      shaderId,
      entryPoint: descriptor.compute.entryPoint,
      label: descriptor.label,
      layout: descriptor.layout === 'auto' ? 'auto' : 'explicit',
      constantCount: Object.keys(constants).length,
      constants,
    };
    this.pipelineIds.set(pipeline, id);
    this.pipelineCaptures.set(id, capture);
    this.events.push({ type: 'pipeline', phase: this.phase, ...capture });
  }

  private captureBindGroup(bindGroup: GPUBindGroup, descriptor: GPUBindGroupDescriptor): void {
    const id = `bind-group-${this.nextBindGroupId++}`;
    const entries = [...descriptor.entries].map((entry) => {
      const resource = entry.resource;
      if ('buffer' in resource) {
        return {
          binding: entry.binding,
          bufferId: this.bufferIds.get(resource.buffer),
          offset: resource.offset,
          size: resource.size,
        };
      }
      return { binding: entry.binding };
    });
    const capture: BindGroupCapture = { id, label: descriptor.label, entries };
    this.bindGroupIds.set(bindGroup, id);
    this.bindGroupCaptures.set(id, capture);
    this.events.push({ type: 'bindGroup', phase: this.phase, ...capture });
  }

  private patchBufferPrototype(buffer: GPUBuffer): void {
    // WebGPU methods require their branded `this`; keep tracer state in a closure.
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const trace = this;
    this.patchPrototype(buffer, 'mapAsync', (original) => {
      const originalMethod = original as GPUBuffer['mapAsync'];
      return async function (
        this: GPUBuffer,
        mode: GPUMapModeFlags,
        offset?: number,
        size?: number,
      ): Promise<undefined> {
        trace.increment('bufferMapAsyncCalls');
        const start = now();
        const promise = originalMethod.call(this, mode, offset, size);
        void promise.then(
          () => {
            const timestamp = now();
            trace.increment('bufferMapAsyncWaitMilliseconds', timestamp - start);
            trace.events.push({
              type: 'mapAsyncResolved',
              phase: trace.phase,
              timestamp,
              bufferId: trace.bufferIds.get(this),
            });
          },
          () => {
            const timestamp = now();
            trace.increment('bufferMapAsyncWaitMilliseconds', timestamp - start);
            trace.events.push({
              type: 'mapAsyncResolved',
              phase: trace.phase,
              timestamp,
              bufferId: trace.bufferIds.get(this),
            });
          },
        );
        return promise;
      } as GPUBuffer['mapAsync'];
    });
    this.patchPrototype(buffer, 'getMappedRange', (original) => {
      const originalMethod = original as GPUBuffer['getMappedRange'];
      return function (this: GPUBuffer, offset?: number, size?: number): ArrayBuffer {
        trace.increment('bufferGetMappedRangeCalls');
        if (size !== undefined) {
          trace.increment('bufferGetMappedRangeRequestedBytes', size);
        }
        return originalMethod.call(this, offset, size);
      } as GPUBuffer['getMappedRange'];
    });
  }

  private patchComputePassPrototype(pass: GPUComputePassEncoder): void {
    // WebGPU methods require their branded `this`; keep tracer state in a closure.
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const trace = this;
    this.patchPrototype(pass, 'setPipeline', (original) => {
      const originalMethod = original as GPUComputePassEncoder['setPipeline'];
      return function (this: GPUComputePassEncoder, pipeline: GPUComputePipeline): undefined {
        trace.increment('computePassSetPipelineCalls');
        const state = trace.passStates.get(this);
        if (state) {
          state.pipelineId = trace.pipelineIds.get(pipeline);
        }
        originalMethod.call(this, pipeline);
      } as GPUComputePassEncoder['setPipeline'];
    });
    this.patchPrototype(pass, 'setBindGroup', (original) => {
      const originalMethod = original as GPUComputePassEncoder['setBindGroup'];
      return function (
        this: GPUComputePassEncoder,
        index: number,
        bindGroup: GPUBindGroup | null,
        dynamicOffsets?: Iterable<number>,
      ): undefined {
        trace.increment('computePassSetBindGroupCalls');
        const state = trace.passStates.get(this);
        const bindGroupId = bindGroup ? trace.bindGroupIds.get(bindGroup) : undefined;
        if (state && bindGroupId) {
          state.bindGroupIds.set(index, bindGroupId);
        }
        originalMethod.call(this, index, bindGroup, dynamicOffsets);
      } as GPUComputePassEncoder['setBindGroup'];
    });
    this.patchPrototype(pass, 'dispatchWorkgroups', (original) => {
      const originalMethod = original as GPUComputePassEncoder['dispatchWorkgroups'];
      return function (this: GPUComputePassEncoder, x: number, y = 1, z = 1): undefined {
        trace.increment('dispatchWorkgroupsCalls');
        const state = trace.passStates.get(this);
        trace.events.push({
          type: 'dispatch',
          phase: trace.phase,
          passId: state?.id ?? 'unknown-pass',
          pipelineId: state?.pipelineId,
          x,
          y,
          z,
          bindGroupIds: state
            ? [...state.bindGroupIds].sort(([lhs], [rhs]) => lhs - rhs).map(([, id]) => id)
            : undefined,
        });
        originalMethod.call(this, x, y, z);
      } as GPUComputePassEncoder['dispatchWorkgroups'];
    });
    this.patchPrototype(pass, 'dispatchWorkgroupsIndirect', (original) => {
      const originalMethod = original as GPUComputePassEncoder['dispatchWorkgroupsIndirect'];
      return function (this: GPUComputePassEncoder, indirectBuffer: GPUBuffer, indirectOffset: number): undefined {
        trace.increment('dispatchWorkgroupsIndirectCalls');
        const state = trace.passStates.get(this);
        trace.events.push({
          type: 'dispatch',
          phase: trace.phase,
          passId: state?.id ?? 'unknown-pass',
          pipelineId: state?.pipelineId,
          indirectBufferId: trace.bufferIds.get(indirectBuffer),
          indirectOffset,
          bindGroupIds: state
            ? [...state.bindGroupIds].sort(([lhs], [rhs]) => lhs - rhs).map(([, id]) => id)
            : undefined,
        });
        originalMethod.call(this, indirectBuffer, indirectOffset);
      } as GPUComputePassEncoder['dispatchWorkgroupsIndirect'];
    });
    this.patchPrototype(pass, 'end', (original) => {
      const originalMethod = original as GPUComputePassEncoder['end'];
      return function (this: GPUComputePassEncoder): undefined {
        trace.increment('computePassEndCalls');
        originalMethod.call(this);
      } as GPUComputePassEncoder['end'];
    });
  }

  private patchCommandEncoderPrototype(encoder: GPUCommandEncoder): void {
    // WebGPU methods require their branded `this`; keep tracer state in a closure.
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const trace = this;
    this.patchPrototype(encoder, 'beginComputePass', (original) => {
      const originalMethod = original as GPUCommandEncoder['beginComputePass'];
      return function (this: GPUCommandEncoder, descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder {
        trace.increment('beginComputePassCalls');
        if (descriptor?.timestampWrites) {
          trace.increment('computePassesWithTimestampWrites');
        }
        const pass = originalMethod.call(this, descriptor);
        trace.passStates.set(pass, { id: `pass-${trace.nextPassId++}`, bindGroupIds: new Map() });
        trace.patchComputePassPrototype(pass);
        return pass;
      } as GPUCommandEncoder['beginComputePass'];
    });
    this.patchPrototype(encoder, 'copyBufferToBuffer', (original) => {
      const originalMethod = original as GPUCommandEncoder['copyBufferToBuffer'];
      return function (
        this: GPUCommandEncoder,
        source: GPUBuffer,
        sourceOffset: number,
        destination: GPUBuffer,
        destinationOffset: number,
        size: number,
      ): undefined {
        trace.increment('copyBufferToBufferCalls');
        trace.increment('copyBufferToBufferBytes', size);
        trace.events.push({
          type: 'copyBufferToBuffer',
          phase: trace.phase,
          sourceId: trace.bufferIds.get(source),
          destinationId: trace.bufferIds.get(destination),
          size,
        });
        originalMethod.call(this, source, sourceOffset, destination, destinationOffset, size);
      } as GPUCommandEncoder['copyBufferToBuffer'];
    });
    this.patchPrototype(encoder, 'finish', (original) => {
      const originalMethod = original as GPUCommandEncoder['finish'];
      return function (this: GPUCommandEncoder, descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer {
        trace.increment('commandEncoderFinishCalls');
        return originalMethod.call(this, descriptor);
      } as GPUCommandEncoder['finish'];
    });
  }

  install(): void {
    if (this.installed) {
      throw new Error('WebGPU API trace is already installed.');
    }
    this.installed = true;
    // WebGPU methods require their branded `this`; keep tracer state in a closure.
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const trace = this;
    const device = this.device;

    this.patchPrototype(device, 'createBuffer', (original) => {
      const originalMethod = original as GPUDevice['createBuffer'];
      return function (this: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
        trace.increment('createBufferCalls');
        trace.increment('createBufferBytes', descriptor.size);
        const buffer = trace.timed('createBufferCpuMilliseconds', () => originalMethod.call(this, descriptor));
        trace.captureBuffer(buffer, descriptor);
        return buffer;
      } as GPUDevice['createBuffer'];
    });
    this.patchPrototype(device, 'createShaderModule', (original) => {
      const originalMethod = original as GPUDevice['createShaderModule'];
      return function (this: GPUDevice, descriptor: GPUShaderModuleDescriptor): GPUShaderModule {
        trace.increment('createShaderModuleCalls');
        const module = trace.timed('createShaderModuleCpuMilliseconds', () => originalMethod.call(this, descriptor));
        trace.captureShader(module, descriptor);
        return module;
      } as GPUDevice['createShaderModule'];
    });
    this.patchPrototype(device, 'createComputePipeline', (original) => {
      const originalMethod = original as GPUDevice['createComputePipeline'];
      return function (this: GPUDevice, descriptor: GPUComputePipelineDescriptor): GPUComputePipeline {
        trace.increment('createComputePipelineCalls');
        const pipeline = trace.timed('createComputePipelineCpuMilliseconds', () =>
          originalMethod.call(this, descriptor),
        );
        trace.capturePipeline(pipeline, descriptor);
        return pipeline;
      } as GPUDevice['createComputePipeline'];
    });
    this.patchPrototype(device, 'createComputePipelineAsync', (original) => {
      const originalMethod = original as GPUDevice['createComputePipelineAsync'];
      return async function (this: GPUDevice, descriptor: GPUComputePipelineDescriptor): Promise<GPUComputePipeline> {
        trace.increment('createComputePipelineAsyncCalls');
        const start = now();
        const promise = originalMethod.call(this, descriptor);
        void promise.then(
          (pipeline: GPUComputePipeline) => {
            trace.increment('createComputePipelineAsyncWaitMilliseconds', now() - start);
            trace.capturePipeline(pipeline, descriptor);
          },
          () => trace.increment('createComputePipelineAsyncWaitMilliseconds', now() - start),
        );
        return promise;
      } as GPUDevice['createComputePipelineAsync'];
    });
    this.patchPrototype(device, 'createBindGroup', (original) => {
      const originalMethod = original as GPUDevice['createBindGroup'];
      return function (this: GPUDevice, descriptor: GPUBindGroupDescriptor): GPUBindGroup {
        trace.increment('createBindGroupCalls');
        trace.increment('createBindGroupEntries', [...descriptor.entries].length);
        const bindGroup = trace.timed('createBindGroupCpuMilliseconds', () => originalMethod.call(this, descriptor));
        trace.captureBindGroup(bindGroup, descriptor);
        return bindGroup;
      } as GPUDevice['createBindGroup'];
    });
    this.patchPrototype(device, 'createBindGroupLayout', (original) => {
      const originalMethod = original as GPUDevice['createBindGroupLayout'];
      return function (this: GPUDevice, descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout {
        trace.increment('createBindGroupLayoutCalls');
        trace.increment('createBindGroupLayoutEntries', [...descriptor.entries].length);
        return trace.timed('createBindGroupLayoutCpuMilliseconds', () => originalMethod.call(this, descriptor));
      } as GPUDevice['createBindGroupLayout'];
    });
    this.patchPrototype(device, 'createPipelineLayout', (original) => {
      const originalMethod = original as GPUDevice['createPipelineLayout'];
      return function (this: GPUDevice, descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout {
        trace.increment('createPipelineLayoutCalls');
        return trace.timed('createPipelineLayoutCpuMilliseconds', () => originalMethod.call(this, descriptor));
      } as GPUDevice['createPipelineLayout'];
    });
    this.patchPrototype(device, 'createCommandEncoder', (original) => {
      const originalMethod = original as GPUDevice['createCommandEncoder'];
      return function (this: GPUDevice, descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder {
        trace.increment('createCommandEncoderCalls');
        const encoder = trace.timed('createCommandEncoderCpuMilliseconds', () => originalMethod.call(this, descriptor));
        trace.encoderIds.set(encoder, `encoder-${trace.nextEncoderId++}`);
        trace.patchCommandEncoderPrototype(encoder);
        return encoder;
      } as GPUDevice['createCommandEncoder'];
    });
    this.patchPrototype(device, 'createQuerySet', (original) => {
      const originalMethod = original as GPUDevice['createQuerySet'];
      return function (this: GPUDevice, descriptor: GPUQuerySetDescriptor): GPUQuerySet {
        trace.increment('createQuerySetCalls');
        return originalMethod.call(this, descriptor);
      } as GPUDevice['createQuerySet'];
    });

    const queue = device.queue;
    this.patchPrototype(queue, 'writeBuffer', (original) => {
      const originalMethod = original as GPUQueue['writeBuffer'];
      return function (
        this: GPUQueue,
        buffer: GPUBuffer,
        bufferOffset: number,
        data: AllowSharedBufferSource,
        dataOffset?: number,
        size?: number,
      ): undefined {
        trace.increment('queueWriteBufferCalls');
        const dataBytes = ArrayBuffer.isView(data) ? data.byteLength : data.byteLength;
        trace.increment('queueWriteBufferBytes', size ?? Math.max(0, dataBytes - (dataOffset ?? 0)));
        trace.timed('queueWriteBufferCpuMilliseconds', () =>
          originalMethod.call(this, buffer, bufferOffset, data, dataOffset, size),
        );
      } as GPUQueue['writeBuffer'];
    });
    this.patchPrototype(queue, 'submit', (original) => {
      const originalMethod = original as GPUQueue['submit'];
      return function (this: GPUQueue, commandBuffers: Iterable<GPUCommandBuffer>): undefined {
        const buffers = [...commandBuffers];
        trace.increment('queueSubmitCalls');
        trace.increment('queueSubmittedCommandBuffers', buffers.length);
        trace.events.push({
          type: 'submit',
          phase: trace.phase,
          timestamp: now(),
          commandBufferCount: buffers.length,
        });
        trace.timed('queueSubmitCpuMilliseconds', () => originalMethod.call(this, buffers));
      } as GPUQueue['submit'];
    });
    this.patchPrototype(queue, 'onSubmittedWorkDone', (original) => {
      const originalMethod = original as GPUQueue['onSubmittedWorkDone'];
      return async function (this: GPUQueue): Promise<undefined> {
        trace.increment('queueOnSubmittedWorkDoneCalls');
        const start = now();
        const promise = originalMethod.call(this);
        void promise.then(
          () => trace.increment('queueOnSubmittedWorkDoneWaitMilliseconds', now() - start),
          () => trace.increment('queueOnSubmittedWorkDoneWaitMilliseconds', now() - start),
        );
        return promise;
      } as GPUQueue['onSubmittedWorkDone'];
    });
  }

  setPhase(phase: string): void {
    this.phase = phase;
  }

  mark(): WebGpuTraceMarker {
    return { eventIndex: this.events.length, counters: { ...this.counters } };
  }

  timelineSince(marker: WebGpuTraceMarker): {
    readonly firstSubmitTimestamp?: number;
    readonly lastSubmitTimestamp?: number;
    readonly lastMapAsyncResolutionTimestamp?: number;
    readonly submitCount: number;
  } {
    const events = this.events.slice(marker.eventIndex);
    const submits = events.filter(
      (event): event is Extract<TraceEvent, { readonly type: 'submit' }> => event.type === 'submit',
    );
    const mapResolutions = events.filter(
      (event): event is Extract<TraceEvent, { readonly type: 'mapAsyncResolved' }> => event.type === 'mapAsyncResolved',
    );
    return {
      firstSubmitTimestamp: submits[0]?.timestamp,
      lastSubmitTimestamp: submits[submits.length - 1]?.timestamp,
      lastMapAsyncResolutionTimestamp: mapResolutions[mapResolutions.length - 1]?.timestamp,
      submitCount: submits.length,
    };
  }

  async summarizeSince(
    marker: WebGpuTraceMarker,
    options: { readonly includeDispatchSequence?: boolean; readonly includeShaderMetadata?: boolean } = {},
  ): Promise<Record<string, unknown>> {
    const counterDelta = Object.fromEntries(
      Object.entries(this.counters)
        .map(([name, value]) => [name, value - (marker.counters[name] ?? 0)] as const)
        .filter(([, value]) => value !== 0),
    );
    const events = this.events.slice(marker.eventIndex);
    const dispatches = events.filter(
      (event): event is Extract<TraceEvent, { readonly type: 'dispatch' }> => event.type === 'dispatch',
    );
    const perPipeline = new Map<
      string,
      { calls: number; totalX: number; totalY: number; totalZ: number; indirectCalls: number }
    >();
    for (const dispatch of dispatches) {
      const pipelineId = dispatch.pipelineId ?? 'unknown-pipeline';
      const aggregate = perPipeline.get(pipelineId) ?? {
        calls: 0,
        totalX: 0,
        totalY: 0,
        totalZ: 0,
        indirectCalls: 0,
      };
      aggregate.calls++;
      aggregate.totalX += dispatch.x ?? 0;
      aggregate.totalY += dispatch.y ?? 0;
      aggregate.totalZ += dispatch.z ?? 0;
      aggregate.indirectCalls += dispatch.indirectBufferId ? 1 : 0;
      perPipeline.set(pipelineId, aggregate);
    }

    const buffers = events.filter(
      (event): event is Extract<TraceEvent, { readonly type: 'buffer' }> => event.type === 'buffer',
    );
    const buffersByUsage = new Map<string, { count: number; bytes: number }>();
    for (const buffer of buffers) {
      const key = buffer.usageNames.join('|') || String(buffer.usage);
      const aggregate = buffersByUsage.get(key) ?? { count: 0, bytes: 0 };
      aggregate.count++;
      aggregate.bytes += buffer.size;
      buffersByUsage.set(key, aggregate);
    }

    const pipelineIds = new Set(
      events
        .filter((event): event is Extract<TraceEvent, { readonly type: 'pipeline' }> => event.type === 'pipeline')
        .map((event) => event.id),
    );
    const shaderIds = new Set(
      events
        .filter((event): event is Extract<TraceEvent, { readonly type: 'shader' }> => event.type === 'shader')
        .map((event) => event.id),
    );
    const bindGroupIds = new Set(
      events
        .filter((event): event is Extract<TraceEvent, { readonly type: 'bindGroup' }> => event.type === 'bindGroup')
        .map((event) => event.id),
    );
    const bufferIds = new Set(buffers.map((buffer) => buffer.id));

    return {
      counters: counterDelta,
      dispatchCount: dispatches.length,
      computePassCount: new Set(dispatches.map((dispatch) => dispatch.passId)).size,
      perPipeline: [...perPipeline].map(([pipelineId, aggregate]) => ({ pipelineId, ...aggregate })),
      dispatchSequence: options.includeDispatchSequence
        ? dispatches.map(({ pipelineId, x, y, z, indirectBufferId, indirectOffset, bindGroupIds }) => ({
            pipelineId,
            x,
            y,
            z,
            indirectBufferId,
            indirectOffset,
            bindGroupIds,
          }))
        : undefined,
      buffers: [...bufferIds]
        .map((id) => this.bufferCaptures.get(id))
        .filter((capture): capture is BufferCapture => capture !== undefined),
      bindGroups: [...bindGroupIds]
        .map((id) => this.bindGroupCaptures.get(id))
        .filter((capture): capture is BindGroupCapture => capture !== undefined),
      bufferCreationsByUsage: [...buffersByUsage].map(([usage, aggregate]) => ({ usage, ...aggregate })),
      largestBufferCreations: [...buffers]
        .sort((lhs, rhs) => rhs.size - lhs.size)
        .slice(0, 32)
        .map(({ id, size, usageNames, label }) => ({ id, size, usageNames, label })),
      copies: events
        .filter(
          (event): event is Extract<TraceEvent, { readonly type: 'copyBufferToBuffer' }> =>
            event.type === 'copyBufferToBuffer',
        )
        .map(({ sourceId, destinationId, size }) => ({ sourceId, destinationId, size })),
      pipelines: [...pipelineIds]
        .map((id) => this.pipelineCaptures.get(id))
        .filter((capture): capture is PipelineCapture => capture !== undefined),
      shaders: options.includeShaderMetadata
        ? await Promise.all(
            [...shaderIds]
              .map((id) => this.shaderCaptures.get(id))
              .filter((capture): capture is ShaderCapture => capture !== undefined)
              .map(shaderMetadata),
          )
        : undefined,
    };
  }

  restore(): void {
    for (let index = this.restores.length - 1; index >= 0; index--) {
      this.restores[index]();
    }
    this.restores.length = 0;
    this.shaderCaptures.clear();
    this.pipelineCaptures.clear();
    this.bufferCaptures.clear();
    this.bindGroupCaptures.clear();
    this.installed = false;
  }
}

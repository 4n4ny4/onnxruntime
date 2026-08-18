// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as esbuild from 'esbuild';
import minimist from 'minimist';
import { execFileSync } from 'node:child_process';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

const args = minimist(process.argv.slice(2), {
  boolean: ['serve', 'smoke'],
});
const ORT_ENGINE = (args['ort-engine'] ?? process.env.npm_config_ort_engine ?? 'native-webgpu-ep') as
  | 'jsep-webgpu'
  | 'native-webgpu-ep';

const WEB_ROOT = path.resolve(__dirname, '..');
const SOURCE_ROOT = path.join(WEB_ROOT, 'test', 'benchmark', 'yolo26-webgpu');
const OUTPUT_ROOT = path.join(SOURCE_ROOT, 'dist');
const ORT_BUNDLE_NAME = 'dist/ort.all.min.js';
const ORT_BUNDLE_PATH = path.join(WEB_ROOT, ORT_BUNDLE_NAME);
const LITERT_VERSION = '2.5.3';
const LITERT_WASM_SOURCE = path.join(WEB_ROOT, 'node_modules', '@litertjs', 'core', 'wasm');
const LITERT_WASM_OUTPUT = path.join(OUTPUT_ROOT, 'litert-wasm');

const requiredOrtFiles =
  ORT_ENGINE === 'jsep-webgpu'
    ? ([
        'dist/ort.all.min.js',
        'dist/ort-wasm-simd-threaded.jsep.mjs',
        'dist/ort-wasm-simd-threaded.jsep.wasm',
      ] as const)
    : ([
        'dist/ort.all.min.js',
        'dist/ort-wasm-simd-threaded.asyncify.mjs',
        'dist/ort-wasm-simd-threaded.asyncify.wasm',
      ] as const);

const verifyOrtWebGpuBundle = async (): Promise<void> => {
  for (const relativePath of requiredOrtFiles) {
    try {
      await fs.access(path.join(WEB_ROOT, relativePath));
    } catch {
      throw new Error(
        `Missing ${relativePath}. Pull/build WASM artifacts, then run ` +
          '`npm run build -- --bundle-mode=perf --webgpu-ep` before building the benchmark.',
      );
    }
  }

  const bundle = await fs.readFile(ORT_BUNDLE_PATH, 'utf8');
  const usesNativeWebGpuEp = bundle.includes('ort-wasm-simd-threaded.asyncify');
  const usesJsep = bundle.includes('ort-wasm-simd-threaded.jsep');
  if (
    (ORT_ENGINE === 'jsep-webgpu' && (!usesJsep || usesNativeWebGpuEp)) ||
    (ORT_ENGINE === 'native-webgpu-ep' && (!usesNativeWebGpuEp || usesJsep))
  ) {
    throw new Error(`dist/ort.all.min.js is not the selected ${ORT_ENGINE} build.`);
  }
};

const copyLiteRtWasm = async (): Promise<void> => {
  try {
    await fs.access(LITERT_WASM_SOURCE);
  } catch {
    throw new Error(
      `Missing @litertjs/core@${LITERT_VERSION}. Run \`npm ci\` in js/web before building the benchmark.`,
    );
  }
  await fs.mkdir(OUTPUT_ROOT, { recursive: true });
  await fs.cp(LITERT_WASM_SOURCE, LITERT_WASM_OUTPUT, { recursive: true, force: true });
};

const getGitCommit = (): string =>
  execFileSync('git', ['rev-parse', 'HEAD'], { cwd: WEB_ROOT, encoding: 'utf8' }).trim();

const buildOptions = (): esbuild.BuildOptions => ({
  absWorkingDir: WEB_ROOT,
  bundle: true,
  define: {
    BENCHMARK_BUILD_INFO: JSON.stringify({
      gitCommit: getGitCommit(),
      litertVersion: LITERT_VERSION,
      ortBundle: ORT_BUNDLE_NAME,
      ortEngine: ORT_ENGINE,
    }),
  },
  entryPoints: [path.join(SOURCE_ROOT, 'app.ts')],
  format: 'esm',
  legalComments: 'none',
  logLevel: 'info',
  outfile: path.join(OUTPUT_ROOT, 'app.js'),
  platform: 'browser',
  sourcemap: true,
  target: ['es2022'],
});

const smokePaths = [
  '/test/benchmark/yolo26-webgpu/index.html',
  '/test/benchmark/yolo26-webgpu/styles.css',
  '/test/benchmark/yolo26-webgpu/dist/app.js',
  '/test/benchmark/yolo26-webgpu/dist/litert-wasm/litert_wasm_internal.js',
  '/test/benchmark/yolo26-webgpu/dist/litert-wasm/litert_wasm_internal.wasm',
  '/dist/ort.all.min.js',
  ...(ORT_ENGINE === 'jsep-webgpu'
    ? ['/dist/ort-wasm-simd-threaded.jsep.mjs', '/dist/ort-wasm-simd-threaded.jsep.wasm']
    : ['/dist/ort-wasm-simd-threaded.asyncify.mjs', '/dist/ort-wasm-simd-threaded.asyncify.wasm']),
] as const;

const runSmokeTest = async (): Promise<void> => {
  const context = await esbuild.context(buildOptions());
  try {
    const server = await context.serve({ host: '127.0.0.1', port: 0, servedir: WEB_ROOT });
    for (const requestPath of smokePaths) {
      const response = await fetch(`http://127.0.0.1:${server.port}${requestPath}`);
      if (!response.ok) {
        throw new Error(`Smoke request ${requestPath} failed: HTTP ${response.status}.`);
      }
      if ((await response.arrayBuffer()).byteLength === 0) {
        throw new Error(`Smoke request ${requestPath} returned an empty body.`);
      }
    }
    console.log(`YOLO26 WebGPU benchmark smoke test loaded ${smokePaths.length} resources.`);
  } finally {
    await context.dispose();
  }
};

const serve = async (): Promise<void> => {
  const context = await esbuild.context(buildOptions());
  await context.watch();
  const server = await context.serve({ host: '127.0.0.1', port: 0, servedir: WEB_ROOT });
  console.log(`YOLO26 WebGPU benchmark: http://127.0.0.1:${server.port}/test/benchmark/yolo26-webgpu/index.html`);
  await new Promise<void>(() => {});
};

const main = async (): Promise<void> => {
  await verifyOrtWebGpuBundle();
  await copyLiteRtWasm();
  if (args.smoke) {
    await runSmokeTest();
  } else if (args.serve) {
    await serve();
  } else {
    await esbuild.build(buildOptions());
  }
};

void main();

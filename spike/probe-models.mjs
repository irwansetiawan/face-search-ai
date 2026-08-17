/**
 * Spike step 0 — can onnxruntime-node load the buffalo_l models, and does CoreML help?
 *
 * Answers two questions and nothing else:
 *   1. What are the exact input/output tensor names and shapes we must produce/consume?
 *   2. Does the CoreML EP actually accelerate these two graphs, or silently fall back to CPU?
 *
 * Run: node spike/probe-models.mjs
 */
import ort from 'onnxruntime-node';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODELS = path.join(__dirname, '..', 'models', 'buffalo_l');

const DET = path.join(MODELS, 'det_10g.onnx');       // SCRFD detector
const REC = path.join(MODELS, 'w600k_r50.onnx');     // ArcFace recognizer

const WARMUP = 3;
const RUNS = 10;

function describe(session, label) {
  console.log(`\n  ${label}`);
  for (const name of session.inputNames) {
    const m = session.inputMetadata?.[name] ?? session.inputMetadata?.find?.(x => x.name === name);
    console.log(`    in   ${name}  ${m ? JSON.stringify(m.shape ?? m.dims ?? '?') : '(shape unavailable)'}`);
  }
  for (const name of session.outputNames) {
    const m = session.outputMetadata?.[name] ?? session.outputMetadata?.find?.(x => x.name === name);
    console.log(`    out  ${name}  ${m ? JSON.stringify(m.shape ?? m.dims ?? '?') : '(shape unavailable)'}`);
  }
}

function randomTensor(dims) {
  const n = dims.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.random() * 2 - 1;
  return new ort.Tensor('float32', data, dims);
}

async function bench(modelPath, providers, inputDims, label) {
  let session;
  const createStart = performance.now();
  try {
    session = await ort.InferenceSession.create(modelPath, { executionProviders: providers });
  } catch (e) {
    console.log(`    ${label.padEnd(8)} FAILED TO CREATE: ${e.message}`);
    return null;
  }
  const createMs = performance.now() - createStart;

  const feeds = { [session.inputNames[0]]: randomTensor(inputDims) };

  // Warm up — CoreML compiles the graph on first run; timing that would be misleading.
  for (let i = 0; i < WARMUP; i++) await session.run(feeds);

  const times = [];
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now();
    await session.run(feeds);
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  console.log(
    `    ${label.padEnd(8)} median ${median.toFixed(1).padStart(7)} ms   ` +
    `(min ${times[0].toFixed(1)}, max ${times[times.length - 1].toFixed(1)})   ` +
    `session create ${createMs.toFixed(0)} ms`
  );
  return { median, session };
}

console.log(`onnxruntime-node ${ort.env?.versions?.common ?? '(version unknown)'}`);

// ---- 1. Shapes -------------------------------------------------------------
console.log('\n=== Model I/O ===');
const detSession = await ort.InferenceSession.create(DET, { executionProviders: ['cpu'] });
describe(detSession, 'det_10g.onnx (SCRFD detector)');
const recSession = await ort.InferenceSession.create(REC, { executionProviders: ['cpu'] });
describe(recSession, 'w600k_r50.onnx (ArcFace recognizer)');

// ---- 2. EP comparison ------------------------------------------------------
console.log('\n=== CoreML vs CPU ===');
console.log('\n  det_10g.onnx  @ 1x3x640x640');
const detCpu = await bench(DET, ['cpu'], [1, 3, 640, 640], 'cpu');
const detCoreml = await bench(DET, ['coreml', 'cpu'], [1, 3, 640, 640], 'coreml');

console.log('\n  w600k_r50.onnx  @ 1x3x112x112');
const recCpu = await bench(REC, ['cpu'], [1, 3, 112, 112], 'cpu');
const recCoreml = await bench(REC, ['coreml', 'cpu'], [1, 3, 112, 112], 'coreml');

// ---- 3. Verdict ------------------------------------------------------------
console.log('\n=== Verdict ===');
for (const [name, cpu, coreml] of [
  ['det_10g  ', detCpu, detCoreml],
  ['w600k_r50', recCpu, recCoreml],
]) {
  if (!cpu || !coreml) { console.log(`  ${name}  inconclusive (a provider failed)`); continue; }
  const speedup = cpu.median / coreml.median;
  const verdict = speedup > 1.15 ? 'CoreML wins'
    : speedup < 0.87 ? 'CPU wins — CoreML is a regression'
    : 'no meaningful difference (likely CPU fallback)';
  console.log(`  ${name}  ${speedup.toFixed(2)}x  → ${verdict}`);
}

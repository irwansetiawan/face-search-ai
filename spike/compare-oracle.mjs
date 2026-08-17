/**
 * Spike — diff the TypeScript pipeline against the Python insightface oracle.
 *
 * The test that matters: for the same image, does our 512-d embedding point in the
 * same direction as the reference one? Anything below ~0.99 means a real defect in
 * the port (letterbox interpolation, anchor decode, alignment, normalization) even
 * if end-to-end matching still "looks fine".
 *
 *   node spike/compare-oracle.mjs <oracle.json>
 */
import { readFile } from 'fs/promises';
import { createMatcher, cosine } from './insightface.mjs';

const oraclePath = process.argv[2];
if (!oraclePath) {
  console.error('usage: node spike/compare-oracle.mjs <oracle.json>');
  process.exit(2);
}

// The oracle runs CPU fp32. Pass --cpu to take CoreML (fp16 on the ANE) out of the
// comparison, isolating precision differences from genuine port defects.
const providers = process.argv.includes('--cpu') ? ['cpu'] : ['coreml', 'cpu'];

const oracle = JSON.parse(await readFile(oraclePath, 'utf8'));
const matcher = await createMatcher({ providers });
console.log(`ts providers: ${JSON.stringify(providers)}   oracle: CPU fp32\n`);

console.log('file                      faces(ts/py)   bbox drift px   cosine(ts,py)');
console.log('-'.repeat(76));

let worst = 1;
for (const [file, ref] of Object.entries(oracle)) {
  const { faces } = await matcher.embedAll(file);
  if (faces.length === 0 || !ref.embedding) {
    console.log(`${file.padEnd(40)} no face on one side`);
    continue;
  }
  const largest = faces.reduce((a, b) => (a.width * a.height >= b.width * b.height ? a : b));

  const sim = cosine(largest.embedding, Float32Array.from(ref.embedding));
  const drift = Math.max(
    ...largest.bbox.map((v, i) => Math.abs(v - ref.bbox[i]))
  );
  worst = Math.min(worst, sim);

  const name = file.split('/').pop();
  console.log(
    `${name.padEnd(26)}${String(faces.length).padStart(4)}/${String(ref.n).padEnd(6)}` +
    `${drift.toFixed(1).padStart(12)}   ${sim.toFixed(6).padStart(12)}`
  );
}

console.log('-'.repeat(76));
console.log(`worst embedding agreement: ${worst.toFixed(6)}`);
console.log(
  worst >= 0.999 ? 'VERDICT: port matches the reference implementation.'
  : worst >= 0.99 ? 'VERDICT: close, but there is a small systematic difference worth finding.'
  : 'VERDICT: the port has a real defect — do not trust these embeddings.'
);

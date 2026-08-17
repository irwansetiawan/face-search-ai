/**
 * Spike — does the TS port ever disagree with the Python oracle about a *decision*?
 *
 * Raw embedding agreement (cosine between ts and py vectors for the same face) is a
 * sensitive but indirect measure. What actually matters is whether the pairwise
 * similarities used for match/no-match come out the same. Two implementations whose
 * embeddings sit at 0.99 to each other can still agree perfectly on every verdict,
 * because the same small rotation is applied to both sides of every comparison.
 *
 *   node spike/decision-diff.mjs <oracle.json>
 */
import { readFile } from 'fs/promises';
import { createMatcher, cosine } from './insightface.mjs';

const oraclePath = process.argv[2];
if (!oraclePath) {
  console.error('usage: node spike/decision-diff.mjs <oracle.json>');
  process.exit(2);
}

const oracle = JSON.parse(await readFile(oraclePath, 'utf8'));
const matcher = await createMatcher();

const files = Object.keys(oracle).filter(f => oracle[f].embedding);
const ts = {}, py = {};
for (const f of files) {
  const { faces } = await matcher.embedAll(f);
  ts[f] = faces.reduce((a, b) => (a.width * a.height >= b.width * b.height ? a : b)).embedding;
  py[f] = Float32Array.from(oracle[f].embedding);
}

console.log('pairwise cosine of the largest face in each image\n');
console.log(`${''.padEnd(22)}${'ts'.padStart(9)}${'python'.padStart(9)}${'delta'.padStart(9)}`);
console.log('-'.repeat(49));

let worstDelta = 0;
for (let i = 0; i < files.length; i++) {
  for (let j = i + 1; j < files.length; j++) {
    const a = files[i].split('/').pop().replace('.jpg', '');
    const b = files[j].split('/').pop().replace('.jpg', '');
    const simTs = cosine(ts[files[i]], ts[files[j]]);
    const simPy = cosine(py[files[i]], py[files[j]]);
    const delta = Math.abs(simTs - simPy);
    worstDelta = Math.max(worstDelta, delta);
    console.log(
      `${(a + ' / ' + b).slice(0, 21).padEnd(22)}` +
      `${simTs.toFixed(4).padStart(9)}${simPy.toFixed(4).padStart(9)}${delta.toFixed(4).padStart(9)}`
    );
  }
}

console.log('-'.repeat(49));
console.log(`largest disagreement on any pair: ${worstDelta.toFixed(4)}`);
console.log(
  worstDelta < 0.02
    ? 'VERDICT: decisions are equivalent — no threshold in a sane range would flip.'
    : 'VERDICT: pairwise scores diverge enough to change verdicts. Investigate.'
);

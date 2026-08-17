/**
 * Spike — diff the TypeScript pipeline against the Python insightface oracle.
 *
 * The test that matters: for the same image, does our 512-d embedding point in the
 * same direction as the reference one? Anything below the gate below means a real
 * defect in the port (letterbox interpolation, anchor decode, alignment,
 * normalization) even if end-to-end matching still "looks fine".
 *
 *   node spike/compare-oracle.mjs <oracle.json>
 *
 * Gate: 0.98 (see AGREEMENT_GATE below). Calibrated against the current
 * spike/images/ fixture set, last measured as:
 *
 *   biden_portrait 0.998334 · g8_group 0.999785 · obama_alt 0.999572
 *   obama_cabinet 0.999729 · obama_exif6 0.999769 · obama_portrait 0.992114
 *   obama_rot_notag 0.985194 · obama_young 0.999165
 *
 * `obama_rot_notag.jpg` is a deliberate negative control, not a marginal
 * fixture: pixels rotated 90 degrees with the EXIF orientation tag stripped,
 * so the face is sideways and detection has to work harder for it -- a
 * genuinely harder case where landmarks sit less stably and the known cv2
 * fixed-point-resize residual is amplified. It is expected to be the floor
 * of this table, not a sign of drift. The historically real defect on this
 * project (the pre-port letterbox bug) measured 0.94 -- the 0.98 gate still
 * catches that class of regression with enormous margin. If a future run
 * moves any number in the table above noticeably, especially on a fixture
 * other than obama_rot_notag, that's the actual signal to chase.
 */
const AGREEMENT_GATE = 0.98;
import { readFile } from 'fs/promises';
import { createMatcher, cosine } from './insightface.mjs';

const oraclePath = process.argv[2];
if (!oraclePath) {
  console.error('usage: node spike/compare-oracle.mjs <oracle.json>');
  process.exit(2);
}

// insightface.mjs's createMatcher ignores any providers option -- the real
// src/face/matcher.ts hardcodes ['coreml', 'cpu'] and takes no arguments, by
// design (Task 3's controller ruling). A `--cpu` flag here used to be
// accepted and printed as if it changed which execution provider ran, which
// it never did -- reporting a configuration this script was not actually
// running. There is currently no way to isolate CoreML fp16 precision from
// a genuine port defect via this script; if that's needed, extend
// createMatcher to actually forward providers rather than resurrecting a
// flag that lies about what ran.
const oracle = JSON.parse(await readFile(oraclePath, 'utf8'));
const matcher = await createMatcher();
console.log('oracle: CPU fp32\n');

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
  : worst >= AGREEMENT_GATE ? 'VERDICT: acceptable agreement, including known harder fixtures (e.g. the sideways-face control).'
  : 'VERDICT: the port has a real defect — do not trust these embeddings.'
);

// The spec makes this a binding "done when" criterion, not just a printed
// number a human is trusted to read -- so it must be checkable in CI/scripts
// too. The printed verdict and the exit code must agree -- a verdict below
// AGREEMENT_GATE exits non-zero instead of always exiting 0.
if (worst < AGREEMENT_GATE) process.exit(1);

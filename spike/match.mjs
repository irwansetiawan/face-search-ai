/**
 * Spike CLI — is the person in the probe image present in the target image(s)?
 *
 *   node spike/match.mjs --probe a.jpg --target b.jpg [more.jpg ...]
 *   node spike/match.mjs --selftest
 *
 * Prints raw cosine similarity. No threshold is applied or implied — picking one
 * is a calibration exercise against labelled data, not a constant to guess here.
 */
import { createMatcher, cosine, umeyama } from './insightface.mjs';

// ---------------------------------------------------------------- self test

/**
 * Recover a known similarity transform from point correspondences. A wrong 2x2 SVD
 * produces subtly misaligned crops and quietly degraded embeddings, so fail loudly.
 */
function selftest() {
  const src = [[10, 20], [80, 25], [45, 60], [20, 95], [70, 90]];
  const angle = 0.37, scale = 1.8, tx = 12.5, ty = -7.25;
  const dst = src.map(([x, y]) => [
    scale * (Math.cos(angle) * x - Math.sin(angle) * y) + tx,
    scale * (Math.sin(angle) * x + Math.cos(angle) * y) + ty,
  ]);

  const M = umeyama(src, dst);
  const expected = [
    [scale * Math.cos(angle), -scale * Math.sin(angle), tx],
    [scale * Math.sin(angle), scale * Math.cos(angle), ty],
  ];

  let worst = 0;
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(M[r][c] - expected[r][c]));
  }
  console.log('umeyama recovered matrix:');
  console.log('  got     ', M.map(r => r.map(v => v.toFixed(4)).join('\t')).join('\n           '));
  console.log('  expected', expected.map(r => r.map(v => v.toFixed(4)).join('\t')).join('\n           '));
  console.log(`  max abs error: ${worst.toExponential(3)}`);

  // Residual of mapping src through M onto dst.
  let resid = 0;
  for (let i = 0; i < src.length; i++) {
    const [x, y] = src[i];
    const px = M[0][0] * x + M[0][1] * y + M[0][2];
    const py = M[1][0] * x + M[1][1] * y + M[1][2];
    resid = Math.max(resid, Math.hypot(px - dst[i][0], py - dst[i][1]));
  }
  console.log(`  max point residual: ${resid.toExponential(3)}`);

  const ok = worst < 1e-6 && resid < 1e-6;
  console.log(ok ? '\nSELFTEST PASS' : '\nSELFTEST FAIL');
  process.exit(ok ? 0 : 1);
}

// ---------------------------------------------------------------- cli

const argv = process.argv.slice(2);
if (argv.includes('--selftest')) selftest();

function argList(flag) {
  const i = argv.indexOf(flag);
  if (i === -1) return [];
  const out = [];
  for (let j = i + 1; j < argv.length && !argv[j].startsWith('--'); j++) out.push(argv[j]);
  return out;
}

const probePath = argList('--probe')[0];
const targetPaths = argList('--target');
if (!probePath || targetPaths.length === 0) {
  console.error('usage: node spike/match.mjs --probe a.jpg --target b.jpg [c.jpg ...]');
  console.error('       node spike/match.mjs --selftest');
  process.exit(2);
}

const t0 = performance.now();
const matcher = await createMatcher();
console.log(`sessions ready in ${(performance.now() - t0).toFixed(0)} ms\n`);

const probe = await matcher.embedLargest(probePath);
if (!probe) {
  console.error(`no face detected in probe: ${probePath}`);
  process.exit(1);
}
console.log(`probe: ${probePath}`);
console.log(`  det_score ${probe.score.toFixed(3)}  bbox ${probe.width.toFixed(0)}x${probe.height.toFixed(0)} px\n`);

for (const targetPath of targetPaths) {
  const tStart = performance.now();
  const { faces } = await matcher.embedAll(targetPath);
  const ms = performance.now() - tStart;

  const scored = faces
    .map(f => ({ ...f, sim: cosine(probe.embedding, f.embedding) }))
    .sort((a, b) => b.sim - a.sim);

  console.log(`${targetPath}  — ${faces.length} face(s), ${ms.toFixed(0)} ms`);
  for (const f of scored) {
    console.log(
      `    cosine ${f.sim >= 0 ? ' ' : ''}${f.sim.toFixed(4)}` +
      `   det ${f.score.toFixed(3)}   ${f.width.toFixed(0)}x${f.height.toFixed(0)} px`
    );
  }
  console.log();
}

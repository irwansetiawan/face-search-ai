/**
 * Spike — render the aligned 112x112 ArcFace crops that actually get fed to the
 * recognizer, so alignment can be checked by eye.
 *
 * Alignment is the highest-risk part of this port: a wrong similarity transform
 * still yields plausible-looking cosine numbers, just quietly worse ones. A correct
 * crop has the eyes level and near y=52, nose near y=72, mouth corners near y=92,
 * and the face filling the frame.
 *
 *   node spike/dump-crops.mjs spike/images/obama_portrait.jpg [...]
 */
import path from 'path';
import { mkdir, writeFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { createMatcher } from './insightface.mjs';

const files = process.argv.slice(2);
if (files.length === 0) {
  console.error('usage: node spike/dump-crops.mjs <image> [image ...]');
  process.exit(2);
}

const outDir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'crops');
await mkdir(outDir, { recursive: true });

const matcher = await createMatcher();

for (const file of files) {
  const { faces } = await matcher.embedAll(file);
  if (faces.length === 0) { console.log(`${file}: no face`); continue; }

  // Largest face only — same selection rule as the probe.
  const face = faces.reduce((a, b) => (a.width * a.height >= b.width * b.height ? a : b));

  // faceIndex (detection order) is what alignedCropPng needs, not the
  // position of `face` in this sorted-by-area array.
  const png = await matcher.alignedCropPng(file, face.faceIndex);

  const base = path.basename(file, path.extname(file));
  const out = path.join(outDir, `${base}.aligned.png`);
  await writeFile(out, png);

  // Where did the 5 landmarks land after the transform? Should be ~ARCFACE_DST.
  console.log(`${base}: det ${face.score.toFixed(3)}  ${face.width.toFixed(0)}x${face.height.toFixed(0)} px -> ${path.basename(out)}`);
}

console.log(`\nwrote to ${outDir}`);

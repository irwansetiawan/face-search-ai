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
import sharp from 'sharp';
import path from 'path';
import { mkdir } from 'fs/promises';
import { fileURLToPath } from 'url';
import { createMatcher, loadImage, alignCrop } from './insightface.mjs';

const files = process.argv.slice(2);
if (files.length === 0) {
  console.error('usage: node spike/dump-crops.mjs <image> [image ...]');
  process.exit(2);
}

const outDir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'crops');
await mkdir(outDir, { recursive: true });

const matcher = await createMatcher();

for (const file of files) {
  const img = await loadImage(file);
  const { faces } = await matcher.embedAll(file);
  if (faces.length === 0) { console.log(`${file}: no face`); continue; }

  // Largest face only — same selection rule as the probe.
  const face = faces.reduce((a, b) => (a.width * a.height >= b.width * b.height ? a : b));

  // Undo the (x-127.5)/127.5 normalization and convert CHW float -> HWC uint8.
  const chw = alignCrop(img, face.kps);
  const size = 112, plane = size * size;
  const rgb = Buffer.alloc(plane * 3);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      rgb[i * 3 + c] = Math.max(0, Math.min(255, Math.round(chw[c * plane + i] * 127.5 + 127.5)));
    }
  }

  const base = path.basename(file, path.extname(file));
  const out = path.join(outDir, `${base}.aligned.png`);
  await sharp(rgb, { raw: { width: size, height: size, channels: 3 } }).png().toFile(out);

  // Where did the 5 landmarks land after the transform? Should be ~ARCFACE_DST.
  console.log(`${base}: det ${face.score.toFixed(3)}  ${face.width.toFixed(0)}x${face.height.toFixed(0)} px -> ${path.basename(out)}`);
}

console.log(`\nwrote to ${outDir}`);

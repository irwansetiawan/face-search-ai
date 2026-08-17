/**
 * Spike shim — re-exports the real pipeline from `src/face`, built to `dist/`.
 *
 * This module used to be a private copy of the buffalo_l pipeline (detector
 * forward pass, anchor decode, alignment, ArcFace). That copy has since been
 * ported to `src/face/pipeline.ts` (numerics) and `src/face/matcher.ts`
 * (ONNX sessions + serialized inference). This module now owns no ONNX
 * session and no model path of its own — every number here comes from
 * `dist/face/pipeline.js` and `dist/face/matcher.js`, so `compare-oracle.mjs`
 * is a genuine regression check on the app's own code, not on a spike-only
 * fork of it.
 *
 * Because this imports from `dist/`, run `npm run build:node` (or `npm test`)
 * before using any script that imports this module — a stale or missing
 * `dist/` will fail with a module-not-found error.
 *
 * `createMatcher`'s shape here intentionally differs from `src/face/matcher.ts`:
 * the real `Matcher.embedAll`/`embedLargest` return `EmbeddedFace` (relative
 * `box`, no `kps`), while the scripts in this directory (written against the
 * pre-port spike) expect absolute-pixel `bbox`/`width`/`height` and a `score`
 * field. That adaptation happens below rather than in `matcher.ts`, whose
 * signature and return shape are fixed by Task 3.
 */
import { readFile } from 'fs/promises';
import {
  loadImage as pipelineLoadImage,
  alignCrop,
  umeyama,
  cosine,
} from '../dist/face/pipeline.js';
import { createMatcher as createRealMatcher } from '../dist/face/matcher.js';

export { alignCrop, umeyama, cosine };

/** Accepts a file path (the spike scripts' convention) or an already-loaded Buffer. */
export async function loadImage(input) {
  return pipelineLoadImage(typeof input === 'string' ? await readFile(input) : input);
}

async function toBuffer(input) {
  return typeof input === 'string' ? readFile(input) : input;
}

/**
 * Adapts an `EmbeddedFace` (relative `box`, no `kps`) to the absolute-pixel
 * `{ score, bbox, width, height }` shape the pre-port spike scripts expect.
 */
function toSpikeFace(f, imageWidth, imageHeight) {
  const x1 = f.box.left * imageWidth;
  const y1 = f.box.top * imageHeight;
  const width = f.box.width * imageWidth;
  const height = f.box.height * imageHeight;
  return {
    score: f.detScore,
    bbox: [x1, y1, x1 + width, y1 + height],
    width,
    height,
    embedding: f.embedding,
    faceIndex: f.faceIndex,
  };
}

/**
 * `providers` is accepted for source compatibility with the pre-port spike
 * (which used to take a `--cpu` flag on compare-oracle.mjs to isolate
 * CoreML fp16 precision from a genuine port defect) but ignored:
 * `src/face/matcher.ts` fixes execution providers to `['coreml', 'cpu']`
 * and takes no arguments — see Task 3's controller rulings for why that
 * isn't adapted the other way. The `--cpu` flag itself was removed (Task
 * 12 review fix 4): it was printing a providers header that didn't reflect
 * what actually ran, since this function ignores `providers` regardless of
 * what's passed. If CoreML-vs-CPU isolation is needed again, this is where
 * `providers` would need to start being forwarded to `createRealMatcher`.
 */
export async function createMatcher(_opts) {
  const real = await createRealMatcher();

  async function embedAll(input) {
    const buf = await toBuffer(input);
    const { faces, imageWidth, imageHeight } = await real.embedAll(buf);
    return { faces: faces.map(f => toSpikeFace(f, imageWidth, imageHeight)), imageWidth, imageHeight };
  }

  async function embedLargest(input) {
    // Built on embedAll (rather than real.embedLargest) so the image is
    // decoded once, not twice: this shim needs imageWidth/imageHeight to
    // convert the relative box back to the pixel bbox/width/height the
    // pre-port spike scripts expect, and real.embedLargest doesn't expose them.
    const { faces, imageWidth, imageHeight } = await embedAll(input);
    if (faces.length === 0) return null;
    const largest = faces.reduce((a, b) => (a.width * a.height >= b.width * b.height ? a : b));
    return { ...largest, imageWidth, imageHeight };
  }

  async function alignedCropPng(input, faceIndex) {
    return real.alignedCropPng(await toBuffer(input), faceIndex);
  }

  return { embedAll, embedLargest, alignedCropPng };
}

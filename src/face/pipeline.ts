/**
 * buffalo_l face embedding pipeline in pure Node.
 *
 * Ports the parts of the Python `insightface` package that have no TS equivalent:
 *   letterbox -> SCRFD forward -> anchor decode -> NMS
 *   -> 5-point similarity-transform alignment to 112x112 -> ArcFace -> L2 normalize
 *
 * Deliberate correspondences to the Python (get these wrong and embeddings are
 * quietly bad rather than obviously broken):
 *   - detector normalizes (x - 127.5) / 128.0   ... blobFromImage(1/128,  mean 127.5)
 *   - recognizer normalizes (x - 127.5) / 127.5 ... blobFromImage(1/127.5, mean 127.5)
 *   - both expect RGB, NCHW
 *   - EXIF rotation must be applied before anything else (sharp .rotate())
 *
 * This module owns no ONNX session and no model file paths — Task 3 runs the
 * models and calls decodeDetections with the raw output arrays.
 */
import sharp from 'sharp';

const DET_SIZE = 640;
const DET_THRESH = 0.5;
const NMS_THRESH = 0.4;
const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;

// ArcFace canonical 5-point reference for a 112x112 crop.
const ARCFACE_DST = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

// ---------------------------------------------------------------- types

export type Box = { top: number; left: number; width: number; height: number };

export type RawImage = { data: Buffer; width: number; height: number };

export type DetectedFace = {
  detScore: number;
  bbox: [number, number, number, number];
  kps: number[][];
  widthPx: number;
  heightPx: number;
};

// ---------------------------------------------------------------- image io

/** Decode to raw RGB, honouring EXIF orientation. Rekognition did this for us. */
export async function loadImage(input: Buffer | string): Promise<RawImage> {
  const { data, info } = await sharp(input)
    .rotate()          // applies EXIF orientation, then strips it
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}

/**
 * Resize preserving aspect ratio into the top-left corner of a `size` square.
 *
 * Mirrors insightface's SCRFD.detect exactly, because small deviations here shift
 * the detected landmarks, which shifts the alignment, which quietly degrades the
 * embedding:
 *   - target dimensions use int() truncation, not rounding
 *   - det_scale is new_height/orig_height (not the min-ratio used to pick the size)
 *   - resampling is bilinear with OpenCV's half-pixel centre convention
 *   - padding is uint8 0 *before* normalizing, i.e. -127.5/128
 */
export function letterbox(img: RawImage, size: number): { input: Float32Array; scale: number } {
  const imRatio = img.height / img.width;
  let nw, nh;
  if (imRatio > 1) {           // model_ratio is 1 for a square det size
    nh = size;
    nw = Math.trunc(nh / imRatio);
  } else {
    nw = size;
    nh = Math.trunc(nw * imRatio);
  }
  const scale = nh / img.height;

  const plane = size * size;
  const input = new Float32Array(3 * plane).fill(-127.5 / 128.0);

  const sx = img.width / nw;
  const sy = img.height / nh;
  for (let y = 0; y < nh; y++) {
    // OpenCV INTER_LINEAR maps dst -> src as (d + 0.5) * scale - 0.5.
    let fy = (y + 0.5) * sy - 0.5;
    if (fy < 0) fy = 0;
    const y0 = Math.min(img.height - 1, Math.floor(fy));
    const y1 = Math.min(img.height - 1, y0 + 1);
    const wy = fy - y0;

    for (let x = 0; x < nw; x++) {
      let fx = (x + 0.5) * sx - 0.5;
      if (fx < 0) fx = 0;
      const x0 = Math.min(img.width - 1, Math.floor(fx));
      const x1 = Math.min(img.width - 1, x0 + 1);
      const wx = fx - x0;

      const p00 = (y0 * img.width + x0) * 3;
      const p10 = (y0 * img.width + x1) * 3;
      const p01 = (y1 * img.width + x0) * 3;
      const p11 = (y1 * img.width + x1) * 3;
      const d = y * size + x;

      for (let c = 0; c < 3; c++) {
        const top = img.data[p00 + c] * (1 - wx) + img.data[p10 + c] * wx;
        const bot = img.data[p01 + c] * (1 - wx) + img.data[p11 + c] * wx;
        input[c * plane + d] = ((top * (1 - wy) + bot * wy) - 127.5) / 128.0;
      }
    }
  }
  return { input, scale };
}

// ---------------------------------------------------------------- detection

const anchorCache = new Map<string, Float32Array>();
function anchorCenters(height: number, width: number, stride: number): Float32Array {
  const key = `${height}:${width}:${stride}`;
  const hit = anchorCache.get(key);
  if (hit) return hit;
  const out = new Float32Array(height * width * NUM_ANCHORS * 2);
  let i = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let a = 0; a < NUM_ANCHORS; a++) {
        out[i++] = x * stride;
        out[i++] = y * stride;
      }
    }
  }
  anchorCache.set(key, out);
  return out;
}

type Candidate = { detScore: number; bbox: [number, number, number, number]; kps: number[][] };

function nms(faces: Candidate[], thresh: number): Candidate[] {
  const order = faces.map((_, i) => i).sort((a, b) => faces[b].detScore - faces[a].detScore);
  const keep: Candidate[] = [];
  const dead = new Uint8Array(faces.length);
  for (const i of order) {
    if (dead[i]) continue;
    keep.push(faces[i]);
    const [ax1, ay1, ax2, ay2] = faces[i].bbox;
    const areaA = (ax2 - ax1 + 1) * (ay2 - ay1 + 1);
    for (const j of order) {
      if (j === i || dead[j]) continue;
      const [bx1, by1, bx2, by2] = faces[j].bbox;
      const w = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1) + 1);
      const h = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1) + 1);
      if (w <= 0 || h <= 0) continue;
      const inter = w * h;
      const areaB = (bx2 - bx1 + 1) * (by2 - by1 + 1);
      if (inter / (areaA + areaB - inter) > thresh) dead[j] = 1;
    }
  }
  return keep;
}

/**
 * Decode the nine SCRFD output arrays (scores x3, bboxes x3, kps x3 — one per
 * FPN stride, in session output order) plus the letterbox scale into
 * NMS-filtered faces. Does not run the model — the caller owns the ORT session.
 */
export function decodeDetections(outputs: Float32Array[], scale: number): DetectedFace[] {
  // Output order is scores[3], bboxes[3], kps[3] — one per FPN stride.
  const candidates: Candidate[] = [];
  for (let s = 0; s < STRIDES.length; s++) {
    const stride = STRIDES[s];
    const scores = outputs[s];
    const bboxes = outputs[s + 3];
    const kpss = outputs[s + 6];
    const gh = Math.ceil(DET_SIZE / stride);
    const gw = Math.ceil(DET_SIZE / stride);
    const centers = anchorCenters(gh, gw, stride);

    for (let i = 0; i < scores.length; i++) {
      const score = scores[i];
      if (score < DET_THRESH) continue;
      const cx = centers[i * 2];
      const cy = centers[i * 2 + 1];

      // distance2bbox — predictions are distances from the anchor centre, in stride units.
      const bbox: [number, number, number, number] = [
        (cx - bboxes[i * 4] * stride) / scale,
        (cy - bboxes[i * 4 + 1] * stride) / scale,
        (cx + bboxes[i * 4 + 2] * stride) / scale,
        (cy + bboxes[i * 4 + 3] * stride) / scale,
      ];

      const kps: number[][] = [];
      for (let k = 0; k < 5; k++) {
        kps.push([
          (cx + kpss[i * 10 + k * 2] * stride) / scale,
          (cy + kpss[i * 10 + k * 2 + 1] * stride) / scale,
        ]);
      }
      candidates.push({ detScore: score, bbox, kps });
    }
  }
  const kept = nms(candidates, NMS_THRESH);
  return kept.map(f => ({
    detScore: f.detScore,
    bbox: f.bbox,
    kps: f.kps,
    widthPx: f.bbox[2] - f.bbox[0],
    heightPx: f.bbox[3] - f.bbox[1],
  }));
}

// ---------------------------------------------------------------- alignment

/**
 * Least-squares similarity transform (rotation + uniform scale + translation)
 * mapping src onto dst. Returns [[a,b,tx],[c,d,ty]] — the same thing skimage's
 * SimilarityTransform.estimate produces inside the Python.
 *
 * Uses the exact closed-form 2D Procrustes solution rather than a general
 * Umeyama/SVD. For 2D the minimiser is analytic, so this avoids the sign and
 * reflection-guard subtleties of a hand-rolled 2x2 SVD entirely. It always
 * returns a proper rotation (never a reflection), which is what we want for
 * facial landmarks — a mirrored fit would be wrong regardless of residual.
 */
export function umeyama(src: number[][], dst: number[][]): number[][] {
  const n = src.length;
  const mean = (pts: number[][]) => pts.reduce((a, p) => [a[0] + p[0] / n, a[1] + p[1] / n], [0, 0]);
  const [msx, msy] = mean(src);
  const [mdx, mdy] = mean(dst);

  // a accumulates the aligned (dot) component, b the perpendicular (cross) one.
  let a = 0, b = 0, varSrc = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - msx, sy = src[i][1] - msy;
    const dx = dst[i][0] - mdx, dy = dst[i][1] - mdy;
    a += sx * dx + sy * dy;
    b += sx * dy - sy * dx;
    varSrc += sx * sx + sy * sy;
  }

  // scale*cos(theta) and scale*sin(theta) fall straight out.
  const sc = varSrc > 0 ? a / varSrc : 1;
  const ss = varSrc > 0 ? b / varSrc : 0;

  return [
    [sc, -ss, mdx - (sc * msx - ss * msy)],
    [ss, sc, mdy - (ss * msx + sc * msy)],
  ];
}

/** Warp a face into a 112x112 ArcFace crop, bilinearly sampling the source. */
export function alignCrop(img: RawImage, kps: number[][], size = 112): Float32Array {
  const M = umeyama(kps, ARCFACE_DST);

  // Invert the 2x3 affine so we can sample source pixels per destination pixel.
  const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  const i00 = M[1][1] / det, i01 = -M[0][1] / det;
  const i10 = -M[1][0] / det, i11 = M[0][0] / det;
  const itx = -(i00 * M[0][2] + i01 * M[1][2]);
  const ity = -(i10 * M[0][2] + i11 * M[1][2]);

  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const srcX = i00 * x + i01 * y + itx;
      const srcY = i10 * x + i11 * y + ity;
      const d = y * size + x;

      const x0 = Math.floor(srcX), y0 = Math.floor(srcY);
      if (x0 < 0 || y0 < 0 || x0 + 1 >= img.width || y0 + 1 >= img.height) {
        out[d] = out[plane + d] = out[2 * plane + d] = -1;
        continue;
      }
      const fx = srcX - x0, fy = srcY - y0;
      const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy, w11 = fx * fy;
      const p00 = (y0 * img.width + x0) * 3;
      const p10 = p00 + 3;
      const p01 = p00 + img.width * 3;
      const p11 = p01 + 3;

      for (let c = 0; c < 3; c++) {
        const v = img.data[p00 + c] * w00 + img.data[p10 + c] * w10
                + img.data[p01 + c] * w01 + img.data[p11 + c] * w11;
        out[c * plane + d] = (v - 127.5) / 127.5;   // note: /127.5, not /128
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------- public api

export function toRelativeBox(
  bbox: [number, number, number, number], w: number, h: number
): Box {
  return {
    left: bbox[0] / w,
    top: bbox[1] / h,
    width: (bbox[2] - bbox[0]) / w,
    height: (bbox[3] - bbox[1]) / h,
  };
}

export function cosine(a: Float32Array | number[], b: Float32Array | number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

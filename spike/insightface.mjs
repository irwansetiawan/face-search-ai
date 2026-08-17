/**
 * Spike — buffalo_l face embedding pipeline in pure Node.
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
 */
import ort from 'onnxruntime-node';
import sharp from 'sharp';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, '..', 'models', 'buffalo_l');

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

// ---------------------------------------------------------------- image io

/** Decode to raw RGB, honouring EXIF orientation. Rekognition did this for us. */
export async function loadImage(file) {
  const { data, info } = await sharp(file)
    .rotate()          // applies EXIF orientation, then strips it
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}

/**
 * Resize preserving aspect ratio into the top-left corner of a DET_SIZE square.
 *
 * Mirrors insightface's SCRFD.detect exactly, because small deviations here shift
 * the detected landmarks, which shifts the alignment, which quietly degrades the
 * embedding:
 *   - target dimensions use int() truncation, not rounding
 *   - det_scale is new_height/orig_height (not the min-ratio used to pick the size)
 *   - resampling is bilinear with OpenCV's half-pixel centre convention
 *   - padding is uint8 0 *before* normalizing, i.e. -127.5/128
 */
function letterbox(img, size) {
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

const anchorCache = new Map();
function anchorCenters(height, width, stride) {
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

function nms(faces, thresh) {
  const order = faces.map((_, i) => i).sort((a, b) => faces[b].score - faces[a].score);
  const keep = [];
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

async function detect(detSession, img) {
  const { input, scale } = letterbox(img, DET_SIZE);
  const feeds = {
    [detSession.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, DET_SIZE, DET_SIZE]),
  };
  const out = await detSession.run(feeds);
  const o = detSession.outputNames.map(n => out[n].data);

  // Output order is scores[3], bboxes[3], kps[3] — one per FPN stride.
  const candidates = [];
  for (let s = 0; s < STRIDES.length; s++) {
    const stride = STRIDES[s];
    const scores = o[s];
    const bboxes = o[s + 3];
    const kpss = o[s + 6];
    const gh = Math.ceil(DET_SIZE / stride);
    const gw = Math.ceil(DET_SIZE / stride);
    const centers = anchorCenters(gh, gw, stride);

    for (let i = 0; i < scores.length; i++) {
      const score = scores[i];
      if (score < DET_THRESH) continue;
      const cx = centers[i * 2];
      const cy = centers[i * 2 + 1];

      // distance2bbox — predictions are distances from the anchor centre, in stride units.
      const bbox = [
        (cx - bboxes[i * 4] * stride) / scale,
        (cy - bboxes[i * 4 + 1] * stride) / scale,
        (cx + bboxes[i * 4 + 2] * stride) / scale,
        (cy + bboxes[i * 4 + 3] * stride) / scale,
      ];

      const kps = [];
      for (let k = 0; k < 5; k++) {
        kps.push([
          (cx + kpss[i * 10 + k * 2] * stride) / scale,
          (cy + kpss[i * 10 + k * 2 + 1] * stride) / scale,
        ]);
      }
      candidates.push({ score, bbox, kps });
    }
  }
  return nms(candidates, NMS_THRESH);
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
export function umeyama(src, dst) {
  const n = src.length;
  const mean = pts => pts.reduce((a, p) => [a[0] + p[0] / n, a[1] + p[1] / n], [0, 0]);
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
export function alignCrop(img, kps, size = 112) {
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

export async function createMatcher({ providers = ['coreml', 'cpu'] } = {}) {
  const detSession = await ort.InferenceSession.create(
    path.join(MODEL_DIR, 'det_10g.onnx'), { executionProviders: providers });
  const recSession = await ort.InferenceSession.create(
    path.join(MODEL_DIR, 'w600k_r50.onnx'), { executionProviders: providers });

  async function embedFace(img, kps) {
    const crop = alignCrop(img, kps);
    const feeds = {
      [recSession.inputNames[0]]: new ort.Tensor('float32', crop, [1, 3, 112, 112]),
    };
    const out = await recSession.run(feeds);
    const raw = out[recSession.outputNames[0]].data;
    let norm = 0;
    for (const v of raw) norm += v * v;
    norm = Math.sqrt(norm);
    const embedding = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) embedding[i] = raw[i] / norm;
    return embedding;
  }

  /** All faces in an image, each with a normalized 512-d embedding. */
  async function embedAll(file) {
    const img = await loadImage(file);
    const faces = await detect(detSession, img);
    const results = [];
    for (const f of faces) {
      results.push({
        score: f.score,
        bbox: f.bbox,
        kps: f.kps,
        width: f.bbox[2] - f.bbox[0],
        height: f.bbox[3] - f.bbox[1],
        embedding: await embedFace(img, f.kps),
      });
    }
    return { faces: results, imageWidth: img.width, imageHeight: img.height };
  }

  /** The probe: largest face only, matching CompareFaces' source-image behaviour. */
  async function embedLargest(file) {
    const { faces, imageWidth, imageHeight } = await embedAll(file);
    if (faces.length === 0) return null;
    const largest = faces.reduce((a, b) => (a.width * a.height >= b.width * b.height ? a : b));
    return { ...largest, imageWidth, imageHeight };
  }

  return { embedAll, embedLargest, detSession, recSession };
}

export function cosine(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

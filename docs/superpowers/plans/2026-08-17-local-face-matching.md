# Local Face Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AWS Rekognition with a local InsightFace pipeline, and add a persistent library of saved people that can be reused across sessions.

**Architecture:** A pure-Node ONNX pipeline (SCRFD detection → 5-point alignment → ArcFace embedding) runs behind two endpoints: `/probe` embeds a face once, `/search` scores a photo against those embeddings. Saved people persist reference embeddings plus their original photos in a JSON file and the filesystem, so a person can be re-searched later and re-embedded automatically if the pipeline changes.

**Tech Stack:** TypeScript (ESM), Express, `onnxruntime-node` (CoreML EP), `sharp`, `node:test`. No database.

**Spec:** `docs/superpowers/specs/2026-08-17-local-face-matching-design.md`

## Global Constraints

- Node 20+, ESM throughout (`"type": "module"`). Relative imports carry `.js` extensions.
- No new runtime dependencies beyond `onnxruntime-node` and `sharp`. Tests use built-in `node:test` — no test framework dependency.
- ORT execution providers are always `['coreml', 'cpu']`, in that order.
- Detector score floor `0.5`; NMS IoU `0.4`. **No minimum face-size filter** — a 33×43 px face matched correctly at 0.706 in the spike.
- Detector normalizes `(x-127.5)/128`; recognizer normalizes `(x-127.5)/127.5`. These differ and must not be unified.
- EXIF rotation applied via `sharp().rotate()` before any other operation.
- Every bounding box crossing the API is relative 0–1 (`{top,left,width,height}`).
- `FACE_MATCH_THRESHOLD` env var, default `0.4`, provisional and documented as such.
- Embeddings are L2-normalized, so cosine similarity is a plain dot product.
- `spike/compare-oracle.mjs` must continue to report ≥ 0.98 agreement after any pipeline change (gate lowered from 0.99 when three fixtures were added; see the spec's Goal note).

---

### Task 1: Project setup — test runner and model download

**Files:**
- Modify: `tsconfig.json`
- Modify: `package.json`
- Create: `scripts/setup-models.mjs`
- Create: `src/face/models.ts`
- Test: `src/face/models.test.ts`

**Interfaces:**
- Consumes: nothing
- Produces: `MODEL_DIR: string`, `DET_MODEL: string`, `REC_MODEL: string`, `assertModelsPresent(): void` from `src/face/models.ts`

- [ ] **Step 1: Update `tsconfig.json`**

Add `esModuleInterop` so `import ort from 'onnxruntime-node'` type-checks, and stop excluding tests so `*.test.ts` compiles into `dist/`.

```json
{
    "compilerOptions": {
        "target": "es2020",
        "module": "es2020",
        "sourceMap": true,
        "declaration": true,
        "outDir": "./dist",
        "strict": true,
        "moduleResolution": "node",
        "allowSyntheticDefaultImports": true,
        "esModuleInterop": true
    },
    "include": ["src"],
    "exclude": ["src/static", "node_modules"]
}
```

- [ ] **Step 2: Update `package.json` scripts and dependencies**

Remove `@aws-sdk/client-rekognition`. Add the `test` and `setup:models` scripts. `onnxruntime-node` and `sharp` are already installed.

```json
  "scripts": {
    "clean": "rimraf dist/",
    "copyfiles": "copyfiles -u 1 src/**/*.html dist/",
    "build": "npm run build:node && npm run build:browser",
    "build:node": "tsc",
    "build:browser": "webpack && npm run copyfiles",
    "server": "node dist/server.js",
    "setup:models": "node scripts/setup-models.mjs",
    "test": "npm run build:node && node --test dist/"
  }
```

Then run: `npm uninstall @aws-sdk/client-rekognition`

- [ ] **Step 3: Write `src/face/models.ts`**

```typescript
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** Repo-root-relative, resolved from dist/face/ at runtime. */
export const MODEL_DIR = path.join(__dirname, '..', '..', 'models', 'buffalo_l');
export const DET_MODEL = path.join(MODEL_DIR, 'det_10g.onnx');
export const REC_MODEL = path.join(MODEL_DIR, 'w600k_r50.onnx');

export function modelsPresent(): boolean {
    return fs.existsSync(DET_MODEL) && fs.existsSync(REC_MODEL);
}

/** Fail with an actionable message rather than an ORT file-not-found error. */
export function assertModelsPresent(): void {
    if (!modelsPresent()) {
        throw new Error(
            `Face models not found in ${MODEL_DIR}.\n` +
            `Run: npm run setup:models`
        );
    }
}
```

- [ ] **Step 4: Write the failing test**

Create `src/face/models.test.ts`:

```typescript
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import { DET_MODEL, REC_MODEL, modelsPresent, assertModelsPresent } from './models.js';

test('model paths point at the two required onnx files', () => {
    assert.match(DET_MODEL, /buffalo_l\/det_10g\.onnx$/);
    assert.match(REC_MODEL, /buffalo_l\/w600k_r50\.onnx$/);
});

test('modelsPresent reflects what is on disk', () => {
    assert.equal(modelsPresent(), fs.existsSync(DET_MODEL) && fs.existsSync(REC_MODEL));
});

test('assertModelsPresent passes once models are downloaded', () => {
    assert.doesNotThrow(() => assertModelsPresent());
});
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module './models.js'` (nothing compiled yet) or assertion failures.

- [ ] **Step 6: Write `scripts/setup-models.mjs`**

```javascript
/**
 * Downloads the buffalo_l model pack. Not a postinstall hook: a silent 275 MB
 * download on `npm install` is unwelcome, and only two of the five files matter.
 */
import fs from 'fs';
import path from 'path';
import { pipeline } from 'stream/promises';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, '..', 'models', 'buffalo_l');
const URL_ = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip';
const REQUIRED = ['det_10g.onnx', 'w600k_r50.onnx'];

if (REQUIRED.every(f => fs.existsSync(path.join(MODEL_DIR, f)))) {
    console.log(`Models already present in ${MODEL_DIR}`);
    process.exit(0);
}

fs.mkdirSync(MODEL_DIR, { recursive: true });
const zipPath = path.join(MODEL_DIR, '..', 'buffalo_l.zip');

console.log(`Downloading buffalo_l (275 MB) from ${URL_} ...`);
const res = await fetch(URL_);
if (!res.ok) throw new Error(`Download failed: HTTP ${res.status}`);
await pipeline(res.body, fs.createWriteStream(zipPath));

console.log('Extracting ...');
execFileSync('unzip', ['-o', '-q', zipPath, '-d', MODEL_DIR]);
fs.unlinkSync(zipPath);

const missing = REQUIRED.filter(f => !fs.existsSync(path.join(MODEL_DIR, f)));
if (missing.length) throw new Error(`Extraction incomplete, missing: ${missing.join(', ')}`);
console.log(`Models ready in ${MODEL_DIR}`);
```

- [ ] **Step 7: Run setup and the test**

Run: `npm run setup:models && npm test`
Expected: setup reports "Models already present" (they were downloaded during the spike); all three tests PASS.

- [ ] **Step 8: Commit**

```bash
git add tsconfig.json package.json package-lock.json scripts/setup-models.mjs src/face/models.ts src/face/models.test.ts
git commit -m "Add model setup script and node:test runner"
```

---

### Task 2: Port the face pipeline to TypeScript

**Files:**
- Create: `src/face/pipeline.ts`
- Test: `src/face/pipeline.test.ts`

**Interfaces:**
- Consumes: `DET_MODEL`, `REC_MODEL` from Task 1
- Produces (all exported):
  - `type Box = { top: number; left: number; width: number; height: number }` (relative 0–1)
  - `type RawImage = { data: Buffer; width: number; height: number }`
  - `type DetectedFace = { detScore: number; bbox: [number, number, number, number]; kps: number[][]; widthPx: number; heightPx: number }`
  - `loadImage(input: Buffer | string): Promise<RawImage>`
  - `letterbox(img: RawImage, size: number): { input: Float32Array; scale: number }`
  - `decodeDetections(outputs: Float32Array[], scale: number): DetectedFace[]`
  - `umeyama(src: number[][], dst: number[][]): number[][]`
  - `alignCrop(img: RawImage, kps: number[][], size?: number): Float32Array`
  - `toRelativeBox(bbox: [number,number,number,number], w: number, h: number): Box`
  - `cosine(a: Float32Array | number[], b: Float32Array | number[]): number`

**Context:** `spike/insightface.mjs` is a working, oracle-validated implementation. This task ports it, it does not redesign it. Copy the file and apply the changes below.

- [ ] **Step 1: Write the failing test**

Create `src/face/pipeline.test.ts`:

```typescript
import test from 'node:test';
import assert from 'node:assert/strict';
import { umeyama, cosine, toRelativeBox } from './pipeline.js';

test('umeyama recovers a known similarity transform', () => {
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
    for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 3; c++) {
            assert.ok(Math.abs(M[r][c] - expected[r][c]) < 1e-9,
                `M[${r}][${c}] was ${M[r][c]}, expected ${expected[r][c]}`);
        }
    }
});

test('cosine of a unit vector with itself is 1', () => {
    const v = new Float32Array(512).fill(1 / Math.sqrt(512));
    assert.ok(Math.abs(cosine(v, v) - 1) < 1e-6);
});

test('toRelativeBox converts absolute pixels to 0-1', () => {
    const box = toRelativeBox([100, 50, 300, 250], 1000, 500);
    assert.equal(box.left, 0.1);
    assert.equal(box.top, 0.1);
    assert.equal(box.width, 0.2);
    assert.equal(box.height, 0.4);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module './pipeline.js'`.

- [ ] **Step 3: Create `src/face/pipeline.ts` from the spike**

```bash
cp spike/insightface.mjs src/face/pipeline.ts
```

Then apply these changes, and **only** these:

1. Delete the `createMatcher` export entirely — session ownership moves to Task 3.
2. Delete the `MODEL_DIR` / `path` / `fileURLToPath` constants; this module no longer touches model files.
3. Export `letterbox` and `decodeDetections` (the current `detect()` body, minus the `session.run` call — it takes the nine output arrays instead of running the model).
4. Change `loadImage(file)` to `loadImage(input: Buffer | string)`. `sharp()` accepts both, so the body is unchanged; only the signature moves.
5. Add the type annotations listed in **Interfaces** above. Note one rename: the spike's
   detected faces carry `score`; `DetectedFace` calls it **`detScore`**, because Task 3
   and the API responses use that name throughout. Rename it in `decodeDetections`.
6. Add `toRelativeBox`:

```typescript
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
```

`decodeDetections` takes the nine SCRFD output arrays in session output order (scores ×3, bboxes ×3, kps ×3) plus the letterbox scale, and returns NMS-filtered `DetectedFace[]`. Keep the existing anchor decode, `distance2bbox` maths, and NMS untouched.

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm test`
Expected: all three PASS.

- [ ] **Step 5: Verify the spike oracle still agrees**

Point the spike at the ported module so it stays a live regression check:

```bash
# in spike/insightface.mjs, replace the pipeline internals with a re-export:
#   export { loadImage, alignCrop, umeyama, cosine, letterbox, decodeDetections } from '../dist/face/pipeline.js';
node spike/match.mjs --selftest
```
Expected: `SELFTEST PASS`

- [ ] **Step 6: Commit**

```bash
git add src/face/pipeline.ts src/face/pipeline.test.ts spike/insightface.mjs
git commit -m "Port face pipeline from spike to TypeScript"
```

---

### Task 3: Matcher — session ownership and serialized inference

**Files:**
- Create: `src/face/matcher.ts`
- Test: `src/face/matcher.test.ts`

**Interfaces:**
- Consumes: Task 1 model paths, Task 2 pipeline functions
- Produces:
  - `type EmbeddedFace = { detScore: number; box: Box; embedding: Float32Array }`
  - `type EmbedResult = { faces: EmbeddedFace[]; imageWidth: number; imageHeight: number }`
  - `createMatcher(): Promise<Matcher>`
  - `Matcher.embedAll(input: Buffer): Promise<EmbedResult>`
  - `Matcher.embedLargest(input: Buffer): Promise<EmbeddedFace | null>`
  - `Matcher.alignedCropPng(input: Buffer, faceIndex: number): Promise<Buffer>` — index into detection order, not the sorted `faces` array

**Context:** Sessions cost 0.7 s and 4.4 s to create, so they are built once. Inference is serialized because the model is the bottleneck — concurrent `session.run` calls contend without improving throughput.

- [ ] **Step 1: Write the failing test**

Create `src/face/matcher.test.ts`:

```typescript
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import path from 'path';
import { createMatcher } from './matcher.js';
import { cosine } from './pipeline.js';

const IMAGES = path.join(process.cwd(), 'spike', 'images');
const read = (n: string) => fs.readFileSync(path.join(IMAGES, n));

test('matcher embeds and separates known identities', async (t) => {
    const matcher = await createMatcher();

    const probe = await matcher.embedLargest(read('obama_portrait.jpg'));
    assert.ok(probe, 'expected a face in the probe image');
    assert.equal(probe!.embedding.length, 512);

    const same = await matcher.embedLargest(read('obama_alt.jpg'));
    const diff = await matcher.embedLargest(read('biden_portrait.jpg'));

    const simSame = cosine(probe!.embedding, same!.embedding);
    const simDiff = cosine(probe!.embedding, diff!.embedding);

    assert.ok(simSame > 0.6, `same-person cosine was ${simSame}, expected > 0.6`);
    assert.ok(simDiff < 0.2, `different-person cosine was ${simDiff}, expected < 0.2`);
});

test('embedAll finds every face in a crowd and returns relative boxes', async () => {
    const matcher = await createMatcher();
    const { faces } = await matcher.embedAll(read('obama_cabinet.jpg'));

    assert.ok(faces.length >= 20, `expected 20+ faces, got ${faces.length}`);
    for (const f of faces) {
        assert.ok(f.box.left >= 0 && f.box.left <= 1, 'left must be relative');
        assert.ok(f.box.top >= 0 && f.box.top <= 1, 'top must be relative');
        assert.ok(f.detScore >= 0.5, 'detector floor must be applied');
    }
});

test('concurrent calls are serialized and both return correct results', async () => {
    const matcher = await createMatcher();
    const [a, b] = await Promise.all([
        matcher.embedLargest(read('obama_portrait.jpg')),
        matcher.embedLargest(read('obama_alt.jpg')),
    ]);
    assert.ok(cosine(a!.embedding, b!.embedding) > 0.6,
        'interleaved inference must not corrupt results');
});

test('EXIF orientation is applied', async () => {
    const matcher = await createMatcher();
    const upright = await matcher.embedLargest(read('obama_alt.jpg'));
    const rotated = await matcher.embedLargest(read('obama_exif6.jpg'));
    const sim = cosine(upright!.embedding, rotated!.embedding);
    assert.ok(sim > 0.97, `EXIF-rotated copy scored ${sim}, expected > 0.97`);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module './matcher.js'`.

- [ ] **Step 3: Write `src/face/matcher.ts`**

```typescript
import ort from 'onnxruntime-node';
import sharp from 'sharp';
import { DET_MODEL, REC_MODEL, assertModelsPresent } from './models.js';
import {
    loadImage, letterbox, decodeDetections, alignCrop, toRelativeBox,
    type Box, type RawImage,
} from './pipeline.js';

const DET_SIZE = 640;

export type EmbeddedFace = { detScore: number; box: Box; embedding: Float32Array };
export type EmbedResult = { faces: EmbeddedFace[]; imageWidth: number; imageHeight: number };

export type Matcher = {
    embedAll(input: Buffer): Promise<EmbedResult>;
    embedLargest(input: Buffer): Promise<EmbeddedFace | null>;
    alignedCropPng(input: Buffer, faceIndex: number): Promise<Buffer>;
};

export async function createMatcher(): Promise<Matcher> {
    assertModelsPresent();
    const providers = ['coreml', 'cpu'];
    const det = await ort.InferenceSession.create(DET_MODEL, { executionProviders: providers });
    const rec = await ort.InferenceSession.create(REC_MODEL, { executionProviders: providers });

    // The model is the bottleneck; serialize so concurrent requests queue
    // rather than contend.
    let queue: Promise<unknown> = Promise.resolve();
    function serialize<T>(fn: () => Promise<T>): Promise<T> {
        const run = queue.then(fn, fn);
        queue = run.catch(() => {});
        return run;
    }

    async function detectFaces(img: RawImage) {
        const { input, scale } = letterbox(img, DET_SIZE);
        const feeds = {
            [det.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, DET_SIZE, DET_SIZE]),
        };
        const out = await det.run(feeds);
        const arrays = det.outputNames.map(n => out[n].data as Float32Array);
        return decodeDetections(arrays, scale);
    }

    async function embedCrop(img: RawImage, kps: number[][]): Promise<Float32Array> {
        const crop = alignCrop(img, kps);
        const feeds = {
            [rec.inputNames[0]]: new ort.Tensor('float32', crop, [1, 3, 112, 112]),
        };
        const out = await rec.run(feeds);
        const raw = out[rec.outputNames[0]].data as Float32Array;
        let norm = 0;
        for (const v of raw) norm += v * v;
        norm = Math.sqrt(norm);
        const embedding = new Float32Array(raw.length);
        for (let i = 0; i < raw.length; i++) embedding[i] = raw[i] / norm;
        return embedding;
    }

    async function embedAll(input: Buffer): Promise<EmbedResult> {
        return serialize(async () => {
            const img = await loadImage(input);
            const detected = await detectFaces(img);
            const faces: EmbeddedFace[] = [];
            for (const f of detected) {
                faces.push({
                    detScore: f.detScore,
                    box: toRelativeBox(f.bbox, img.width, img.height),
                    embedding: await embedCrop(img, f.kps),
                });
            }
            return { faces, imageWidth: img.width, imageHeight: img.height };
        });
    }

    async function embedLargest(input: Buffer): Promise<EmbeddedFace | null> {
        const { faces } = await embedAll(input);
        if (faces.length === 0) return null;
        return faces.reduce((a, b) =>
            a.box.width * a.box.height >= b.box.width * b.box.height ? a : b);
    }

    /** The aligned 112x112 crop, rendered as PNG. Used as a saved person's avatar. */
    async function alignedCropPng(input: Buffer, faceIndex: number): Promise<Buffer> {
        return serialize(async () => {
            const img = await loadImage(input);
            const detected = await detectFaces(img);
            const f = detected[faceIndex];
            if (!f) throw new Error(`no face at index ${faceIndex}`);
            const chw = alignCrop(img, f.kps);
            const size = 112, plane = size * size;
            const rgb = Buffer.alloc(plane * 3);
            for (let i = 0; i < plane; i++) {
                for (let c = 0; c < 3; c++) {
                    rgb[i * 3 + c] = Math.max(0, Math.min(255,
                        Math.round(chw[c * plane + i] * 127.5 + 127.5)));
                }
            }
            return sharp(rgb, { raw: { width: size, height: size, channels: 3 } })
                .png().toBuffer();
        });
    }

    return { embedAll, embedLargest, alignedCropPng };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm test`
Expected: all four PASS. First run is slow (~5 s per `createMatcher`).

- [ ] **Step 5: Commit**

```bash
git add src/face/matcher.ts src/face/matcher.test.ts
git commit -m "Add matcher with serialized inference and eager sessions"
```

---

### Task 4: POST /probe

**Files:**
- Create: `src/routes/probe.ts`
- Create: `src/app.ts`
- Test: `src/routes/probe.test.ts`

**Interfaces:**
- Consumes: `Matcher` from Task 3
- Produces:
  - `createApp(matcher: Matcher): express.Express` from `src/app.ts`
  - `probeHandler(matcher: Matcher): RequestHandler` from `src/routes/probe.ts`
  - Response: `{ embedding: number[], face: { box: Box, score: number } }`

**Context:** `createApp` takes the matcher as an argument so tests can start the app without going through `server.ts` boot.

- [ ] **Step 1: Write the failing test**

Create `src/routes/probe.test.ts`:

```typescript
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import path from 'path';
import { createMatcher } from '../face/matcher.js';
import { createApp } from '../app.js';

const IMAGES = path.join(process.cwd(), 'spike', 'images');

async function withServer(fn: (base: string) => Promise<void>) {
    const app = createApp(await createMatcher());
    const server = app.listen(0);
    await new Promise(r => server.once('listening', r));
    const { port } = server.address() as { port: number };
    try {
        await fn(`http://127.0.0.1:${port}`);
    } finally {
        server.close();
    }
}

function form(field: string, file: string) {
    const fd = new FormData();
    fd.append(field, new Blob([fs.readFileSync(path.join(IMAGES, file))]), file);
    return fd;
}

test('POST /probe returns a 512-d embedding and a relative box', async () => {
    await withServer(async (base) => {
        const res = await fetch(`${base}/probe`, {
            method: 'POST', body: form('source', 'obama_portrait.jpg'),
        });
        assert.equal(res.status, 200);
        const body = await res.json();
        assert.equal(body.embedding.length, 512);
        assert.ok(body.face.score > 0.5);
        assert.ok(body.face.box.width > 0 && body.face.box.width <= 1);
    });
});

test('POST /probe returns 422 when there is no face', async () => {
    await withServer(async (base) => {
        const blank = await (await import('sharp')).default({
            create: { width: 200, height: 200, channels: 3, background: '#888' },
        }).jpeg().toBuffer();
        const fd = new FormData();
        fd.append('source', new Blob([blank]), 'blank.jpg');

        const res = await fetch(`${base}/probe`, { method: 'POST', body: fd });
        assert.equal(res.status, 422);
        assert.equal((await res.json()).error, 'no_face_detected');
    });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module '../app.js'`.

- [ ] **Step 3: Write `src/routes/probe.ts`**

```typescript
import type { RequestHandler } from 'express';
import fs from 'fs/promises';
import type { Matcher } from '../face/matcher.js';

export function probeHandler(matcher: Matcher): RequestHandler {
    return async (req, res) => {
        const files = req.files as { [f: string]: Express.Multer.File[] };
        const upload = files?.source?.[0];
        if (!upload) {
            res.status(400).json({ error: 'bad_probe' });
            return;
        }
        try {
            const buffer = await fs.readFile(upload.path);
            const face = await matcher.embedLargest(buffer);
            if (!face) {
                res.status(422).json({ error: 'no_face_detected' });
                return;
            }
            res.status(200).json({
                embedding: Array.from(face.embedding),
                face: { box: face.box, score: face.detScore },
            });
        } catch {
            res.status(400).json({ error: 'unreadable_image' });
        } finally {
            // Always clean up; the old code unlinked only on the success path.
            await fs.unlink(upload.path).catch(() => {});
        }
    };
}
```

- [ ] **Step 4: Write `src/app.ts`**

```typescript
import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import type { Matcher } from './face/matcher.js';
import { probeHandler } from './routes/probe.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function createApp(matcher: Matcher): express.Express {
    const app = express();
    const upload = multer({ dest: 'uploads/' });

    app.use(express.static(path.join(__dirname, '/static')));
    app.post('/probe', upload.fields([{ name: 'source', maxCount: 1 }]), probeHandler(matcher));

    return app;
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npm test`
Expected: both probe tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/app.ts src/routes/probe.ts src/routes/probe.test.ts
git commit -m "Add POST /probe endpoint"
```

---

### Task 5: POST /search

**Files:**
- Create: `src/routes/search.ts`
- Create: `src/face/score.ts`
- Modify: `src/app.ts`
- Test: `src/face/score.test.ts`, `src/routes/search.test.ts`

**Interfaces:**
- Consumes: `Matcher`, `createApp`
- Produces:
  - `scoreFaces(faces: EmbeddedFace[], refs: Float32Array[], threshold: number): ScoredFace[]` from `src/face/score.ts`
  - `type ScoredFace = { box: Box; cosine: number; detScore: number; matched: boolean; bestReference: number }`
  - `MATCH_THRESHOLD: number` (from `FACE_MATCH_THRESHOLD`, default 0.4)
  - Response: `{ threshold, matched, faces: ScoredFace[] }`

- [ ] **Step 1: Write the failing test for scoring**

Create `src/face/score.test.ts`:

```typescript
import test from 'node:test';
import assert from 'node:assert/strict';
import { scoreFaces } from './score.js';
import type { EmbeddedFace } from './matcher.js';

function unit(...values: number[]): Float32Array {
    const v = new Float32Array(512);
    values.forEach((x, i) => { v[i] = x; });
    let n = 0; for (const x of v) n += x * x;
    n = Math.sqrt(n);
    for (let i = 0; i < v.length; i++) v[i] /= n;
    return v;
}

const box = { top: 0, left: 0, width: 0.1, height: 0.1 };
const face = (embedding: Float32Array): EmbeddedFace => ({ detScore: 0.9, box, embedding });

test('scores against the best of several references', () => {
    const a = unit(1, 0), b = unit(0, 1);
    const [scored] = scoreFaces([face(b)], [a, b], 0.4);
    assert.ok(scored.cosine > 0.99, `expected ~1 against the matching ref, got ${scored.cosine}`);
    assert.equal(scored.bestReference, 1);
    assert.equal(scored.matched, true);
});

test('a face below threshold is not matched', () => {
    const [scored] = scoreFaces([face(unit(0, 1))], [unit(1, 0)], 0.4);
    assert.ok(scored.cosine < 0.4);
    assert.equal(scored.matched, false);
});

test('results are sorted by descending similarity', () => {
    const ref = unit(1, 0);
    const scored = scoreFaces(
        [face(unit(0, 1)), face(unit(1, 0)), face(unit(1, 1))], [ref], 0.4);
    assert.ok(scored[0].cosine >= scored[1].cosine);
    assert.ok(scored[1].cosine >= scored[2].cosine);
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module './score.js'`.

- [ ] **Step 3: Write `src/face/score.ts`**

```typescript
import { cosine, type Box } from './pipeline.js';
import type { EmbeddedFace } from './matcher.js';

export const MATCH_THRESHOLD = Number(process.env.FACE_MATCH_THRESHOLD ?? 0.4);

export type ScoredFace = {
    box: Box;
    cosine: number;
    detScore: number;
    matched: boolean;
    bestReference: number;
};

/**
 * Score each detected face against every reference embedding, keeping the best.
 * Max-over-references is what absorbs pose and lighting variation: a profile and a
 * front-on reference sit far apart in embedding space, and a candidate only needs
 * to resemble one of them.
 */
export function scoreFaces(
    faces: EmbeddedFace[], refs: Float32Array[], threshold: number
): ScoredFace[] {
    return faces
        .map(f => {
            let best = -Infinity, bestReference = -1;
            refs.forEach((ref, i) => {
                const sim = cosine(f.embedding, ref);
                if (sim > best) { best = sim; bestReference = i; }
            });
            return {
                box: f.box,
                cosine: best,
                detScore: f.detScore,
                matched: best >= threshold,
                bestReference,
            };
        })
        .sort((a, b) => b.cosine - a.cosine);
}
```

- [ ] **Step 4: Run to verify scoring passes**

Run: `npm test`
Expected: three scoring tests PASS.

- [ ] **Step 5: Write the failing route test**

Create `src/routes/search.test.ts`:

```typescript
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import path from 'path';
import { createMatcher } from '../face/matcher.js';
import { createApp } from '../app.js';

const IMAGES = path.join(process.cwd(), 'spike', 'images');

async function withServer(fn: (base: string) => Promise<void>) {
    const app = createApp(await createMatcher());
    const server = app.listen(0);
    await new Promise(r => server.once('listening', r));
    const { port } = server.address() as { port: number };
    try { await fn(`http://127.0.0.1:${port}`); } finally { server.close(); }
}

async function probe(base: string, file: string): Promise<number[]> {
    const fd = new FormData();
    fd.append('source', new Blob([fs.readFileSync(path.join(IMAGES, file))]), file);
    const res = await fetch(`${base}/probe`, { method: 'POST', body: fd });
    return (await res.json()).embedding;
}

async function search(base: string, embedding: number[], file: string) {
    const fd = new FormData();
    fd.append('target', new Blob([fs.readFileSync(path.join(IMAGES, file))]), file);
    fd.append('probe', JSON.stringify([embedding]));
    const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
    return { status: res.status, body: await res.json() };
}

test('finds the probe subject in a crowd and rejects everyone else', async () => {
    await withServer(async (base) => {
        const embedding = await probe(base, 'obama_portrait.jpg');
        const { status, body } = await search(base, embedding, 'obama_cabinet.jpg');

        assert.equal(status, 200);
        assert.equal(body.matched, true);
        assert.equal(body.faces.filter((f: any) => f.matched).length, 1,
            'exactly one face in the cabinet photo should match');
        assert.ok(body.faces.length >= 20);
    });
});

test('reports no match for a photo of someone else', async () => {
    await withServer(async (base) => {
        const embedding = await probe(base, 'obama_portrait.jpg');
        const { body } = await search(base, embedding, 'biden_portrait.jpg');
        assert.equal(body.matched, false);
    });
});

test('a photo with no faces is 200 with an empty list, not an error', async () => {
    await withServer(async (base) => {
        const embedding = await probe(base, 'obama_portrait.jpg');
        const blank = await (await import('sharp')).default({
            create: { width: 200, height: 200, channels: 3, background: '#888' },
        }).jpeg().toBuffer();
        const fd = new FormData();
        fd.append('target', new Blob([blank]), 'blank.jpg');
        fd.append('probe', JSON.stringify([embedding]));

        const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
        assert.equal(res.status, 200);
        const body = await res.json();
        assert.deepEqual(body.faces, []);
        assert.equal(body.matched, false);
    });
});

test('a malformed probe is rejected with 400', async () => {
    await withServer(async (base) => {
        const fd = new FormData();
        fd.append('target', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_alt.jpg'))]), 'x.jpg');
        fd.append('probe', JSON.stringify([[1, 2, 3]]));  // wrong length
        const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
        assert.equal(res.status, 400);
        assert.equal((await res.json()).error, 'bad_probe');
    });
});
```

- [ ] **Step 6: Run to verify it fails**

Run: `npm test`
Expected: FAIL — 404 on `/search`.

- [ ] **Step 7: Write `src/routes/search.ts`**

```typescript
import type { RequestHandler } from 'express';
import fs from 'fs/promises';
import type { Matcher } from '../face/matcher.js';
import { scoreFaces, MATCH_THRESHOLD } from '../face/score.js';

/** Parses the `probe` field: a JSON array of 512-length embedding arrays. */
export function parseProbe(raw: unknown): Float32Array[] | null {
    if (typeof raw !== 'string') return null;
    let parsed: unknown;
    try { parsed = JSON.parse(raw); } catch { return null; }
    if (!Array.isArray(parsed) || parsed.length === 0) return null;
    const refs: Float32Array[] = [];
    for (const v of parsed) {
        if (!Array.isArray(v) || v.length !== 512) return null;
        refs.push(Float32Array.from(v));
    }
    return refs;
}

export function searchHandler(matcher: Matcher): RequestHandler {
    return async (req, res) => {
        const files = req.files as { [f: string]: Express.Multer.File[] };
        const upload = files?.target?.[0];
        if (!upload) {
            res.status(400).json({ error: 'unreadable_image' });
            return;
        }
        const refs = parseProbe(req.body?.probe);
        if (!refs) {
            await fs.unlink(upload.path).catch(() => {});
            res.status(400).json({ error: 'bad_probe' });
            return;
        }
        try {
            const buffer = await fs.readFile(upload.path);
            const { faces } = await matcher.embedAll(buffer);
            const scored = scoreFaces(faces, refs, MATCH_THRESHOLD);
            res.status(200).json({
                threshold: MATCH_THRESHOLD,
                matched: scored.some(f => f.matched),
                faces: scored,
            });
        } catch {
            res.status(400).json({ error: 'unreadable_image' });
        } finally {
            await fs.unlink(upload.path).catch(() => {});
        }
    };
}
```

- [ ] **Step 8: Register the route in `src/app.ts`**

Add the import and the route:

```typescript
import { searchHandler } from './routes/search.js';
// ...
    app.post('/search', upload.fields([{ name: 'target', maxCount: 1 }]), searchHandler(matcher));
```

- [ ] **Step 9: Run to verify all pass**

Run: `npm test`
Expected: all search and score tests PASS.

- [ ] **Step 10: Commit**

```bash
git add src/face/score.ts src/face/score.test.ts src/routes/search.ts src/routes/search.test.ts src/app.ts
git commit -m "Add POST /search with multi-reference scoring"
```

---

### Task 6: Server boot and Rekognition removal

**Files:**
- Modify: `src/server.ts`
- Delete: `src/compare-face.ts`
- Modify: `.env.example`, `README.md`, `.gitignore`

**Interfaces:**
- Consumes: `createApp`, `createMatcher`
- Produces: a runnable server on port 3100

- [ ] **Step 1: Rewrite `src/server.ts`**

```typescript
import 'dotenv/config';
import { createApp } from './app.js';
import { createMatcher } from './face/matcher.js';
import { MATCH_THRESHOLD } from './face/score.js';

const port = 3100;

// Models take ~5s to compile. Load before listening so the first request is not
// an outlier and a missing model pack fails loudly at boot.
console.log('Loading face models (this takes a few seconds) ...');
const started = Date.now();
const matcher = await createMatcher();
console.log(`Models ready in ${((Date.now() - started) / 1000).toFixed(1)}s`);
console.log(`Match threshold: ${MATCH_THRESHOLD} (provisional, set FACE_MATCH_THRESHOLD to change)`);

createApp(matcher).listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
```

- [ ] **Step 2: Delete the Rekognition implementation**

```bash
git rm src/compare-face.ts
```

- [ ] **Step 3: Update `.env.example`**

Replace the AWS keys with:

```
# Provisional. Same-person cosine measured 0.69-0.77; worst impostor 0.18.
# Not a calibrated operating point - see the spec.
FACE_MATCH_THRESHOLD=0.4
```

- [ ] **Step 4: Update `README.md`**

Replace the AWS account requirements section with:

```markdown
# Requirements

1. Node 20+ on macOS (Apple Silicon recommended — CoreML gives 3.6x on detection
   and 55x on recognition).
2. Download the face models once: `npm run setup:models` (275 MB).

No cloud account and no API costs — everything runs locally.

# Run in localhost

```
npm install
npm run setup:models
npm run build
npm run server
```

Then open http://localhost:3100/ in your browser.
```

Also update the opening line, which currently says "powered by AWS".

- [ ] **Step 5: Verify the server boots and answers**

```bash
npm run build && npm run server &
sleep 12
curl -s -F "source=@spike/images/obama_portrait.jpg" http://localhost:3100/probe | head -c 100
kill %1
```
Expected: JSON beginning `{"embedding":[`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Boot local matcher and remove AWS Rekognition"
```

---

### Task 7: Frontend two-phase flow

**Files:**
- Modify: `src/static/index.ts`

**Interfaces:**
- Consumes: `/probe` and `/search`
- Produces: no exports; browser code

**Context:** The progress counters, zip logic, and canvas drawing all stay. What changes is that the probe is fetched once before the loop, and the response field names are ours rather than AWS's.

- [ ] **Step 1: Fix the latent `isDirectory` bug**

At `src/static/index.ts:92`, `if (!isDirectory)` tests the function object, which is always truthy, so the single-image branch is dead and line 94 is unreachable. Change to `if (!isDirectory())`.

- [ ] **Step 2: Replace the submit handler's request flow**

Fetch the probe once, then loop targets. Replace the body of the `submit` listener after the `targetFiles` validation:

```typescript
    // One probe for the whole run - this is the entire point of the split.
    let probe: number[];
    try {
        const probeForm = new FormData();
        probeForm.append('source', sourceFile);
        const probeRes = await fetch('/probe', { method: 'POST', body: probeForm });
        if (probeRes.status === 422) {
            alert('No face found in the source image. Try a clearer photo.');
            return;
        }
        if (!probeRes.ok) { alert('Could not read the source image.'); return; }
        const probeJson = await probeRes.json();
        probe = probeJson.embedding;
        drawSourceBox(probeJson.face.box);
    } catch (e) {
        alert('Could not reach the server.'); return;
    }
```

Then pass `probe` into `sendRequest`, which now posts to `/search`:

```typescript
function sendRequest(probe: number[], targetFile: File): Promise<void> {
    onRequestSent();
    sendingRequest = true;
    return (async () => {
        const body = new FormData();
        body.append('target', targetFile);
        body.append('probe', JSON.stringify([probe]));
        const res = await fetch('/search', { method: 'POST', body });
        onResponseReceived();
        await handleResponse(res, targetFile);
    })();
}
```

- [ ] **Step 3: Update `handleResponse` to the new shape**

The AWS field names are gone. `faces[].box` is already relative, and `matched` is decided server-side:

```typescript
async function handleResponse(res: Response, targetFile: File): Promise<void> {
    const body = await res.json();
    if (!res.ok) { console.warn(targetFile.name, body.error); return; }

    if (isSingle()) {
        const targetCanvas = document.createElement('canvas');
        targetCanvas.id = 'targetCanvas';
        locateElementOnTopOf(targetImg, targetCanvas);
        for (const face of body.faces) {
            canvasRect(targetCanvas, face.box, face.matched ? '#FF0000' : '#888888');
            if (face.matched) {
                canvasRectLabel(targetCanvas, face.cosine.toFixed(3) + ' cosine', face.box);
            }
        }
        return;
    }

    if (body.matched) {
        onFaceMatched();
        filesToBeZipped.push(targetFile);
    }
}
```

Add `drawSourceBox`, replacing the old inline `SourceImageFace` handling:

```typescript
function drawSourceBox(box: RelativeBox) {
    cleanCanvases();
    const sourceCanvas = document.createElement('canvas');
    sourceCanvas.id = 'sourceCanvas';
    locateElementOnTopOf(sourceImg, sourceCanvas);
    canvasRect(sourceCanvas, box, '#FF0000');
}
```

- [ ] **Step 4: Update `RelativeBox` to lowercase keys**

`canvasRect` and `canvasRectLabel` currently read `Top`/`Left`/`Width`/`Height`. Our API uses lowercase. Change the type and both functions:

```typescript
type RelativeBox = { top: number; left: number; width: number; height: number };
```

- [ ] **Step 5: Verify manually in the browser**

```bash
npm run build && npm run server
```

Open http://localhost:3100/ and check:
1. Single image: probe `spike/images/obama_portrait.jpg` against `spike/images/obama_cabinet.jpg`. One red box on Obama, grey boxes on everyone else, cosine label ~0.706.
2. Network tab: exactly **one** `/probe` call and one `/search` call per photo — the source image must not be re-uploaded.
3. Directory mode: point at `spike/images/`, confirm progress advances and a zip downloads containing only the Obama photos.

- [ ] **Step 6: Commit**

```bash
git add src/static/index.ts
git commit -m "Switch frontend to two-phase probe and search"
```

---

### Task 8: People store

**Files:**
- Create: `src/people/store.ts`
- Test: `src/people/store.test.ts`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `type Reference = { id: string; image: string; thumb: string; embedding: number[]; detScore: number; addedAt: string }`
  - `type Person = { id: string; name: string; createdAt: string; references: Reference[] }`
  - `createStore(dataDir: string): Promise<Store>`
  - `Store.list(): Person[]`
  - `Store.get(id: string): Person | undefined`
  - `Store.addPerson(name: string, ref: Omit<Reference,'id'|'addedAt'>): Promise<Person>`
  - `Store.addReference(personId: string, ref: Omit<Reference,'id'|'addedAt'>): Promise<Person>`
  - `Store.deletePerson(id: string): Promise<void>`
  - `Store.deleteReference(personId: string, refId: string): Promise<Person>`
  - `PIPELINE_VERSION = 1`

- [ ] **Step 1: Write the failing test**

Create `src/people/store.test.ts`:

```typescript
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { createStore } from './store.js';

async function tmpDir(): Promise<string> {
    return fs.mkdtemp(path.join(os.tmpdir(), 'people-'));
}

const ref = (fill: number) => ({
    image: 'people/x/ref-1.jpg',
    thumb: 'people/x/ref-1.thumb.png',
    embedding: new Array(512).fill(fill),
    detScore: 0.9,
});

test('adds a person and persists across reloads', async () => {
    const dir = await tmpDir();
    const store = await createStore(dir);
    const person = await store.addPerson('Sarah', ref(0.1));

    assert.equal(person.name, 'Sarah');
    assert.equal(person.references.length, 1);

    const reloaded = await createStore(dir);
    assert.equal(reloaded.list().length, 1);
    assert.equal(reloaded.get(person.id)!.name, 'Sarah');
});

test('supports multiple references per person', async () => {
    const store = await createStore(await tmpDir());
    const person = await store.addPerson('Sarah', ref(0.1));
    const updated = await store.addReference(person.id, ref(0.2));
    assert.equal(updated.references.length, 2);
    assert.notEqual(updated.references[0].id, updated.references[1].id);
});

test('refuses to delete the last reference', async () => {
    const store = await createStore(await tmpDir());
    const person = await store.addPerson('Sarah', ref(0.1));
    await assert.rejects(
        () => store.deleteReference(person.id, person.references[0].id),
        /last reference/
    );
});

test('deletes a person', async () => {
    const dir = await tmpDir();
    const store = await createStore(dir);
    const person = await store.addPerson('Sarah', ref(0.1));
    await store.deletePerson(person.id);
    assert.equal(store.list().length, 0);
    assert.equal((await createStore(dir)).list().length, 0);
});

test('concurrent writes do not lose entries', async () => {
    const dir = await tmpDir();
    const store = await createStore(dir);
    await Promise.all([
        store.addPerson('A', ref(0.1)),
        store.addPerson('B', ref(0.2)),
        store.addPerson('C', ref(0.3)),
    ]);
    assert.equal((await createStore(dir)).list().length, 3);
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module './store.js'`.

- [ ] **Step 3: Write `src/people/store.ts`**

```typescript
import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';

export const PIPELINE_VERSION = 1;

export type Reference = {
    id: string;
    image: string;
    thumb: string;
    embedding: number[];
    detScore: number;
    addedAt: string;
};

export type Person = {
    id: string;
    name: string;
    createdAt: string;
    references: Reference[];
};

type FileShape = { pipelineVersion: number; people: Person[] };

export type Store = {
    list(): Person[];
    get(id: string): Person | undefined;
    addPerson(name: string, ref: NewReference): Promise<Person>;
    addReference(personId: string, ref: NewReference): Promise<Person>;
    deletePerson(id: string): Promise<void>;
    deleteReference(personId: string, refId: string): Promise<Person>;
    replaceAll(people: Person[]): Promise<void>;
    dataDir: string;
};

export type NewReference = Omit<Reference, 'id' | 'addedAt'>;

const shortId = (prefix: string) => `${prefix}_${crypto.randomBytes(3).toString('hex')}`;

export async function createStore(dataDir: string): Promise<Store> {
    const file = path.join(dataDir, 'people.json');
    await fs.mkdir(path.join(dataDir, 'people'), { recursive: true });

    let state: FileShape = { pipelineVersion: PIPELINE_VERSION, people: [] };
    try {
        state = JSON.parse(await fs.readFile(file, 'utf8'));
    } catch {
        // First run: no file yet.
    }

    // Serialize writes so concurrent requests cannot interleave read-modify-write.
    let queue: Promise<unknown> = Promise.resolve();
    function serialize<T>(fn: () => Promise<T>): Promise<T> {
        const run = queue.then(fn, fn);
        queue = run.catch(() => {});
        return run;
    }

    /** Temp file + rename, so a crash mid-write cannot truncate the store. */
    async function flush(): Promise<void> {
        const tmp = `${file}.tmp`;
        await fs.writeFile(tmp, JSON.stringify(state, null, 2));
        await fs.rename(tmp, file);
    }

    function mustGet(id: string): Person {
        const person = state.people.find(p => p.id === id);
        if (!person) throw new Error(`no such person: ${id}`);
        return person;
    }

    return {
        dataDir,
        list: () => state.people,
        get: (id) => state.people.find(p => p.id === id),

        addPerson: (name, ref) => serialize(async () => {
            const person: Person = {
                id: shortId('p'),
                name,
                createdAt: new Date().toISOString(),
                references: [{ ...ref, id: shortId('ref'), addedAt: new Date().toISOString() }],
            };
            state.people.push(person);
            await flush();
            return person;
        }),

        addReference: (personId, ref) => serialize(async () => {
            const person = mustGet(personId);
            person.references.push({
                ...ref, id: shortId('ref'), addedAt: new Date().toISOString(),
            });
            await flush();
            return person;
        }),

        deletePerson: (id) => serialize(async () => {
            state.people = state.people.filter(p => p.id !== id);
            await fs.rm(path.join(dataDir, 'people', id), { recursive: true, force: true });
            await flush();
        }),

        deleteReference: (personId, refId) => serialize(async () => {
            const person = mustGet(personId);
            if (person.references.length <= 1) {
                throw new Error('cannot delete the last reference; delete the person instead');
            }
            person.references = person.references.filter(r => r.id !== refId);
            await flush();
            return person;
        }),

        replaceAll: (people) => serialize(async () => {
            state = { pipelineVersion: PIPELINE_VERSION, people };
            await flush();
        }),
    };
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `npm test`
Expected: all five store tests PASS.

- [ ] **Step 5: Add `data/` to `.gitignore`**

Append `data/` to `.gitignore`.

- [ ] **Step 6: Commit**

```bash
git add src/people/store.ts src/people/store.test.ts .gitignore
git commit -m "Add people store with atomic writes"
```

---

### Task 9: Re-embed saved people when the pipeline changes

**Files:**
- Create: `src/people/reembed.ts`
- Test: `src/people/reembed.test.ts`

**Interfaces:**
- Consumes: `Store`, `PIPELINE_VERSION`, `Matcher`
- Produces: `reembedIfStale(store: Store, matcher: Matcher, storedVersion: number): Promise<number>` — returns how many references were re-embedded

**Context:** During the spike, a letterbox fix changed every embedding it had computed (agreement 0.94 → 0.992). Stored originals exist precisely so this is recoverable without asking the user to re-upload.

- [ ] **Step 1: Write the failing test**

Create `src/people/reembed.test.ts`:

```typescript
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { createStore, PIPELINE_VERSION } from './store.js';
import { reembedIfStale } from './reembed.js';
import { createMatcher } from '../face/matcher.js';

const SRC = path.join(process.cwd(), 'spike', 'images', 'obama_portrait.jpg');

test('stale stored embeddings are recomputed from the stored image', async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'reembed-'));
    const store = await createStore(dir);
    const matcher = await createMatcher();

    const rel = path.join('people', 'tmp', 'ref-1.jpg');
    await fs.mkdir(path.join(dir, 'people', 'tmp'), { recursive: true });
    await fs.copyFile(SRC, path.join(dir, rel));

    // A deliberately wrong embedding, as a stale one would be.
    const person = await store.addPerson('Sarah', {
        image: rel, thumb: rel, embedding: new Array(512).fill(0), detScore: 0.9,
    });

    const count = await reembedIfStale(store, matcher, PIPELINE_VERSION - 1);
    assert.equal(count, 1);

    const fixed = store.get(person.id)!.references[0].embedding;
    const norm = Math.sqrt(fixed.reduce((a, v) => a + v * v, 0));
    assert.ok(Math.abs(norm - 1) < 1e-3, `expected an L2-normalized vector, norm was ${norm}`);
});

test('current-version stores are left alone', async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'reembed-'));
    const store = await createStore(dir);
    const matcher = await createMatcher();
    await store.addPerson('Sarah', {
        image: 'nonexistent.jpg', thumb: 'x.png',
        embedding: new Array(512).fill(0), detScore: 0.9,
    });

    const count = await reembedIfStale(store, matcher, PIPELINE_VERSION);
    assert.equal(count, 0, 'must not touch a current store, even with a bad path');
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module './reembed.js'`.

- [ ] **Step 3: Write `src/people/reembed.ts`**

```typescript
import fs from 'fs/promises';
import path from 'path';
import type { Store, Person } from './store.js';
import { PIPELINE_VERSION } from './store.js';
import type { Matcher } from '../face/matcher.js';

/**
 * Recompute every stored embedding from its original image when the pipeline
 * version has moved on. Returns the number of references updated.
 */
export async function reembedIfStale(
    store: Store, matcher: Matcher, storedVersion: number
): Promise<number> {
    if (storedVersion === PIPELINE_VERSION) return 0;

    let updated = 0;
    const people: Person[] = store.list();
    for (const person of people) {
        for (const ref of person.references) {
            const abs = path.join(store.dataDir, ref.image);
            try {
                const face = await matcher.embedLargest(await fs.readFile(abs));
                if (!face) continue;
                ref.embedding = Array.from(face.embedding);
                ref.detScore = face.detScore;
                updated++;
            } catch {
                // Keep the stale vector rather than dropping the person entirely;
                // a missing file should not delete someone's saved reference.
            }
        }
    }
    if (updated > 0) await store.replaceAll(people);
    return updated;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `npm test`
Expected: both re-embed tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/people/reembed.ts src/people/reembed.test.ts
git commit -m "Re-embed saved people when the pipeline version changes"
```

---

### Task 10: People routes

**Files:**
- Create: `src/routes/people.ts`
- Modify: `src/app.ts`, `src/server.ts`
- Test: `src/routes/people.test.ts`

**Interfaces:**
- Consumes: `Store`, `Matcher`
- Produces: `peopleRouter(store: Store, matcher: Matcher): express.Router`
- `createApp(matcher: Matcher, store: Store)` — signature gains a second argument

- [ ] **Step 1: Write the failing test**

Create `src/routes/people.test.ts`:

```typescript
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import fsp from 'fs/promises';
import os from 'os';
import path from 'path';
import { createMatcher } from '../face/matcher.js';
import { createStore } from '../people/store.js';
import { createApp } from '../app.js';

const IMAGES = path.join(process.cwd(), 'spike', 'images');

async function withServer(fn: (base: string) => Promise<void>) {
    const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'people-routes-'));
    const app = createApp(await createMatcher(), await createStore(dir));
    const server = app.listen(0);
    await new Promise(r => server.once('listening', r));
    const { port } = server.address() as { port: number };
    try { await fn(`http://127.0.0.1:${port}`); } finally { server.close(); }
}

function photoForm(file: string, fields: Record<string, string> = {}) {
    const fd = new FormData();
    fd.append('photo', new Blob([fs.readFileSync(path.join(IMAGES, file))]), file);
    for (const [k, v] of Object.entries(fields)) fd.append(k, v);
    return fd;
}

test('creates a person, lists them, and adds a second reference', async () => {
    await withServer(async (base) => {
        const created = await fetch(`${base}/people`, {
            method: 'POST', body: photoForm('obama_portrait.jpg', { name: 'Barack' }),
        });
        assert.equal(created.status, 201);
        const person = await created.json();
        assert.equal(person.name, 'Barack');
        assert.equal(person.references.length, 1);

        const listed = await (await fetch(`${base}/people`)).json();
        assert.equal(listed.length, 1);
        assert.ok(listed[0].references[0].thumb, 'expected an avatar path');

        const added = await fetch(`${base}/people/${person.id}/references`, {
            method: 'POST', body: photoForm('obama_alt.jpg'),
        });
        assert.equal(added.status, 200);
        assert.equal((await added.json()).references.length, 2);
    });
});

test('rejects a photo with no face', async () => {
    await withServer(async (base) => {
        const blank = await (await import('sharp')).default({
            create: { width: 200, height: 200, channels: 3, background: '#888' },
        }).jpeg().toBuffer();
        const fd = new FormData();
        fd.append('photo', new Blob([blank]), 'blank.jpg');
        fd.append('name', 'Nobody');

        const res = await fetch(`${base}/people`, { method: 'POST', body: fd });
        assert.equal(res.status, 422);
        assert.equal((await res.json()).error, 'no_face_detected');
    });
});

test('deletes a person', async () => {
    await withServer(async (base) => {
        const person = await (await fetch(`${base}/people`, {
            method: 'POST', body: photoForm('obama_portrait.jpg', { name: 'Barack' }),
        })).json();

        const del = await fetch(`${base}/people/${person.id}`, { method: 'DELETE' });
        assert.equal(del.status, 204);
        assert.equal((await (await fetch(`${base}/people`)).json()).length, 0);
    });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — 404 on `/people`.

- [ ] **Step 3: Write `src/routes/people.ts`**

```typescript
import express from 'express';
import multer from 'multer';
import fs from 'fs/promises';
import path from 'path';
import sharp from 'sharp';
import type { Store, NewReference } from '../people/store.js';
import type { Matcher } from '../face/matcher.js';

const MAX_STORED_EDGE = 1600;   // detection runs at 640; larger is storage we cannot use

/** Public shape: never ship 512-float embeddings to the browser's people list. */
function publicPerson(p: import('../people/store.js').Person) {
    return {
        id: p.id, name: p.name, createdAt: p.createdAt,
        references: p.references.map(r => ({
            id: r.id, thumb: r.thumb, detScore: r.detScore, addedAt: r.addedAt,
        })),
    };
}

export function peopleRouter(store: Store, matcher: Matcher): express.Router {
    const router = express.Router();
    const upload = multer({ dest: 'uploads/' });

    /** Embeds, downscales and writes both artefacts. Returns the store record. */
    async function ingest(personId: string, uploadPath: string): Promise<NewReference | null> {
        const buffer = await fs.readFile(uploadPath);
        const face = await matcher.embedLargest(buffer);
        if (!face) return null;

        const refDir = path.join(store.dataDir, 'people', personId);
        await fs.mkdir(refDir, { recursive: true });
        const stamp = Date.now().toString(36);

        const imageRel = path.join('people', personId, `${stamp}.jpg`);
        await sharp(buffer)
            .rotate()                                     // bake in EXIF before storing
            .resize(MAX_STORED_EDGE, MAX_STORED_EDGE, { fit: 'inside', withoutEnlargement: true })
            .jpeg({ quality: 88 })
            .toFile(path.join(store.dataDir, imageRel));

        const thumbRel = path.join('people', personId, `${stamp}.thumb.png`);
        await fs.writeFile(
            path.join(store.dataDir, thumbRel),
            await matcher.alignedCropPng(buffer, 0),
        );

        return {
            image: imageRel, thumb: thumbRel,
            embedding: Array.from(face.embedding), detScore: face.detScore,
        };
    }

    router.get('/', (_req, res) => {
        res.json(store.list().map(publicPerson));
    });

    router.post('/', upload.single('photo'), async (req, res) => {
        const file = req.file;
        const name = String(req.body?.name ?? '').trim();
        if (!file || !name) { res.status(400).json({ error: 'name_and_photo_required' }); return; }
        try {
            const id = `pending_${Date.now().toString(36)}`;
            const ref = await ingest(id, file.path);
            if (!ref) { res.status(422).json({ error: 'no_face_detected' }); return; }
            const person = await store.addPerson(name, ref);
            // ingest wrote under a pending id; move the files under the real one.
            await fs.rename(
                path.join(store.dataDir, 'people', id),
                path.join(store.dataDir, 'people', person.id),
            );
            person.references[0].image = person.references[0].image.replace(id, person.id);
            person.references[0].thumb = person.references[0].thumb.replace(id, person.id);
            await store.replaceAll(store.list());
            res.status(201).json(publicPerson(person));
        } catch {
            res.status(400).json({ error: 'unreadable_image' });
        } finally {
            await fs.unlink(file.path).catch(() => {});
        }
    });

    router.post('/:id/references', upload.single('photo'), async (req, res) => {
        const file = req.file;
        if (!file) { res.status(400).json({ error: 'photo_required' }); return; }
        if (!store.get(req.params.id)) { res.status(404).json({ error: 'no_such_person' }); return; }
        try {
            const ref = await ingest(req.params.id, file.path);
            if (!ref) { res.status(422).json({ error: 'no_face_detected' }); return; }
            res.status(200).json(publicPerson(await store.addReference(req.params.id, ref)));
        } catch {
            res.status(400).json({ error: 'unreadable_image' });
        } finally {
            await fs.unlink(file.path).catch(() => {});
        }
    });

    router.delete('/:id', async (req, res) => {
        if (!store.get(req.params.id)) { res.status(404).json({ error: 'no_such_person' }); return; }
        await store.deletePerson(req.params.id);
        res.status(204).end();
    });

    router.delete('/:id/references/:refId', async (req, res) => {
        try {
            res.json(publicPerson(await store.deleteReference(req.params.id, req.params.refId)));
        } catch (e) {
            res.status(400).json({ error: (e as Error).message });
        }
    });

    return router;
}
```

- [ ] **Step 4: Wire into `src/app.ts` and `src/server.ts`**

`createApp` gains a `store` argument, mounts the router, and serves stored avatars:

```typescript
export function createApp(matcher: Matcher, store: Store): express.Express {
    // ... existing setup ...
    app.use('/people-files', express.static(path.join(store.dataDir)));
    app.use('/people', peopleRouter(store, matcher));
    return app;
}
```

In `src/server.ts`, create the store, run the re-embed check, and pass both:

```typescript
import path from 'path';
import { createStore, PIPELINE_VERSION } from './people/store.js';
import { reembedIfStale } from './people/reembed.js';

const dataDir = path.join(process.cwd(), 'data');
const store = await createStore(dataDir);

const stored = JSON.parse(
    await import('fs/promises').then(fs =>
        fs.readFile(path.join(dataDir, 'people.json'), 'utf8').catch(() => '{}'))
).pipelineVersion ?? PIPELINE_VERSION;

const reembedded = await reembedIfStale(store, matcher, stored);
if (reembedded > 0) console.log(`Re-embedded ${reembedded} saved reference(s) after a pipeline change`);

createApp(matcher, store).listen(port, () => { /* ... */ });
```

Update the existing `probe.test.ts` and `search.test.ts` `createApp` calls to pass a store built on a temp dir.

- [ ] **Step 5: Run to verify all pass**

Run: `npm test`
Expected: all people-route tests PASS, and the earlier probe/search tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/routes/people.ts src/routes/people.test.ts src/app.ts src/server.ts src/routes/probe.test.ts src/routes/search.test.ts
git commit -m "Add people routes with reference photo storage"
```

---

### Task 11: Search by saved person

**Files:**
- Modify: `src/routes/search.ts`, `src/app.ts`
- Test: `src/routes/search.test.ts`

**Interfaces:**
- Consumes: `Store`
- Produces: `/search` accepts `personId` as an alternative to `probe`

**Context:** Three references × 512 floats is ~30 KB per request; across 500 photos that is 15 MB of redundant payload. Saved people are durable stored data, not ephemeral session state, so a server-side lookup does not reintroduce the probe-session lifetime problems that were rejected in design.

- [ ] **Step 1: Write the failing test**

Append to `src/routes/search.test.ts`:

```typescript
test('searches using a saved person id', async () => {
    await withServer(async (base) => {
        const fd = new FormData();
        fd.append('photo', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_portrait.jpg'))]), 'p.jpg');
        fd.append('name', 'Barack');
        const person = await (await fetch(`${base}/people`, { method: 'POST', body: fd })).json();

        const search = new FormData();
        search.append('target', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_cabinet.jpg'))]), 't.jpg');
        search.append('personId', person.id);

        const res = await fetch(`${base}/search`, { method: 'POST', body: search });
        assert.equal(res.status, 200);
        const body = await res.json();
        assert.equal(body.matched, true);
        assert.equal(body.faces.filter((f: any) => f.matched).length, 1);
    });
});

test('an unknown personId is rejected', async () => {
    await withServer(async (base) => {
        const fd = new FormData();
        fd.append('target', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_alt.jpg'))]), 't.jpg');
        fd.append('personId', 'p_nope');
        const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
        assert.equal(res.status, 400);
        assert.equal((await res.json()).error, 'bad_probe');
    });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — 400 `bad_probe` on the saved-person search.

- [ ] **Step 3: Update `searchHandler` to accept a store**

```typescript
export function searchHandler(matcher: Matcher, store: Store): RequestHandler {
    return async (req, res) => {
        // ... existing upload check ...

        let refs: Float32Array[] | null = null;
        const personId = req.body?.personId;
        if (typeof personId === 'string' && personId.length > 0) {
            const person = store.get(personId);
            if (person) refs = person.references.map(r => Float32Array.from(r.embedding));
        } else {
            refs = parseProbe(req.body?.probe);
        }

        if (!refs || refs.length === 0) {
            await fs.unlink(upload.path).catch(() => {});
            res.status(400).json({ error: 'bad_probe' });
            return;
        }
        // ... existing scoring and response ...
    };
}
```

Update the `app.ts` route registration to `searchHandler(matcher, store)`.

- [ ] **Step 4: Run to verify it passes**

Run: `npm test`
Expected: all search tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/routes/search.ts src/routes/search.test.ts src/app.ts
git commit -m "Allow searching by saved person id"
```

---

### Task 12: Saved people in the UI

**Files:**
- Modify: `src/static/index.html`, `src/static/index.ts`

**Interfaces:**
- Consumes: `/people`, `/search` with `personId`
- Produces: no exports

- [ ] **Step 1: Add the source-mode toggle and picker to `index.html`**

Inside the "Select source image" column, above the existing file input:

```html
<div class="mb-2">
    <input type="radio" name="sourceType" value="upload" id="sourceTypeUpload" checked>
    <label for="sourceTypeUpload">Upload a photo</label>
    <input type="radio" name="sourceType" value="person" id="sourceTypePerson">
    <label for="sourceTypePerson">Choose a saved person</label>
</div>
<div id="peoplePicker" style="display:none" class="mb-3"></div>
<div id="saveAsPerson" style="display:none" class="mb-3">
    <input type="text" id="newPersonName" class="form-control mb-1" placeholder="Name">
    <button type="button" id="savePersonBtn" class="btn btn-sm btn-outline-primary">
        Save as person
    </button>
</div>
```

- [ ] **Step 2: Render the picker in `index.ts`**

```typescript
let selectedPersonId: string | null = null;

async function loadPeople(): Promise<void> {
    const people = await (await fetch('/people')).json();
    const picker = document.getElementById('peoplePicker') as HTMLDivElement;
    picker.innerHTML = people.length === 0
        ? '<em>No saved people yet. Upload a photo and save it.</em>'
        : '';
    for (const person of people) {
        const el = document.createElement('div');
        el.className = 'd-inline-block text-center me-2';
        el.innerHTML =
            `<img src="/people-files/${person.references[0].thumb}" width="64" height="64" ` +
            `style="cursor:pointer;border-radius:6px">` +
            `<div style="font-size:12px">${person.name}</div>` +
            `<button type="button" class="btn btn-sm btn-link text-danger p-0">delete</button>`;
        el.querySelector('img')!.addEventListener('click', () => {
            selectedPersonId = person.id;
            document.querySelectorAll('#peoplePicker img')
                .forEach(i => ((i as HTMLElement).style.outline = ''));
            (el.querySelector('img') as HTMLElement).style.outline = '3px solid #0d6efd';
        });
        el.querySelector('button')!.addEventListener('click', async () => {
            if (!confirm(`Delete ${person.name}?`)) return;
            await fetch(`/people/${person.id}`, { method: 'DELETE' });
            if (selectedPersonId === person.id) selectedPersonId = null;
            await loadPeople();
        });
        picker.appendChild(el);
    }
}
```

- [ ] **Step 3: Wire the toggle and the save button**

```typescript
const radioSourceUpload = document.getElementById('sourceTypeUpload') as HTMLInputElement;
const radioSourcePerson = document.getElementById('sourceTypePerson') as HTMLInputElement;

function sourceChanged() {
    const usingPerson = radioSourcePerson.checked;
    (document.getElementById('peoplePicker') as HTMLDivElement).style.display =
        usingPerson ? 'block' : 'none';
    sourceInput.style.display = usingPerson ? 'none' : 'block';
    if (usingPerson) loadPeople();
}
radioSourceUpload.addEventListener('change', sourceChanged);
radioSourcePerson.addEventListener('change', sourceChanged);

document.getElementById('savePersonBtn')!.addEventListener('click', async () => {
    const name = (document.getElementById('newPersonName') as HTMLInputElement).value.trim();
    const file = sourceInput.files?.[0];
    if (!name || !file) { alert('Pick a photo and enter a name.'); return; }
    const fd = new FormData();
    fd.append('photo', file);
    fd.append('name', name);
    const res = await fetch('/people', { method: 'POST', body: fd });
    if (res.status === 422) { alert('No face found in that photo.'); return; }
    if (!res.ok) { alert('Could not save that person.'); return; }
    alert(`Saved ${name}.`);
    (document.getElementById('newPersonName') as HTMLInputElement).value = '';
});
```

Show the save control after a successful probe by adding this to the probe branch from Task 7:

```typescript
        (document.getElementById('saveAsPerson') as HTMLDivElement).style.display = 'block';
```

- [ ] **Step 4: Skip `/probe` when a saved person is selected**

In the submit handler, branch before fetching the probe:

```typescript
    const usingPerson = radioSourcePerson.checked;
    if (usingPerson && !selectedPersonId) { alert('Pick a saved person.'); return; }
    let probe: number[] | null = null;
    if (!usingPerson) {
        // ... existing /probe fetch from Task 7 ...
    }
```

And in `sendRequest`, send whichever identifier applies:

```typescript
        if (selectedPersonId && radioSourcePerson.checked) {
            body.append('personId', selectedPersonId);
        } else {
            body.append('probe', JSON.stringify([probe]));
        }
```

- [ ] **Step 5: Verify manually**

```bash
npm run build && npm run server
```

1. Upload `spike/images/obama_portrait.jpg`, name it "Barack", click **Save as person**.
2. Switch to **Choose a saved person** — the avatar appears (the aligned crop).
3. Select it, pick `spike/images/obama_cabinet.jpg` as target, submit. One face matches.
4. Restart the server. The saved person is still there — this is the whole point.
5. Delete the person; confirm they disappear and `data/people/<id>/` is removed.

- [ ] **Step 6: Commit**

```bash
git add src/static/index.html src/static/index.ts
git commit -m "Add saved people picker to the UI"
```

---

## Final verification

- [ ] `npm test` — all tests pass
- [ ] `node spike/match.mjs --selftest` — PASS
- [ ] Oracle regression (needs the Python venv from `spike/README.md`):
      `node spike/compare-oracle.mjs /tmp/oracle.json` reports worst agreement ≥ 0.98 and exits 0
- [ ] `node spike/decision-diff.mjs /tmp/oracle.json` reports largest pairwise
      disagreement < 0.02 — the check that no match verdict changed
- [ ] `grep -ri rekognition src/ package.json` returns nothing
- [ ] Scanning a folder issues exactly one `/probe` and one `/search` per photo

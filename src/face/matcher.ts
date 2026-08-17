/**
 * Owns the ONNX sessions for the buffalo_l detector and recognizer, and
 * serializes inference through them.
 *
 * `pipeline.ts` is a pure numerical port with no model paths and no ORT
 * dependency; this module is where the models actually get run. Sessions are
 * created once per `createMatcher()` call (~0.7s for the detector, ~4.4s for
 * the recognizer to compile) and reused for every embed call after that.
 *
 * Inference is serialized — not because Node can't have two `session.run`
 * calls in flight, but because the model itself is the bottleneck. Letting
 * concurrent calls interleave against the same session just makes them
 * contend for the same compute rather than actually running in parallel.
 */
import ort from 'onnxruntime-node';
import sharp from 'sharp';
import { DET_MODEL, REC_MODEL, assertModelsPresent } from './models.js';
import {
    DET_SIZE, loadImage, letterbox, decodeDetections, alignCrop, toRelativeBox,
    type Box, type RawImage, type DetectedFace,
} from './pipeline.js';

export type EmbeddedFace = {
    detScore: number;
    box: Box;
    embedding: Float32Array;
    /** Index into detection order (pre-sort), not the position in a sorted array. */
    faceIndex: number;
};
export type EmbedResult = { faces: EmbeddedFace[]; imageWidth: number; imageHeight: number };

export type Matcher = {
    embedAll(input: Buffer): Promise<EmbedResult>;
    embedLargest(input: Buffer): Promise<EmbeddedFace | null>;
    /** The aligned 112x112 crop, rendered as PNG. Used as a saved person's avatar. */
    alignedCropPng(input: Buffer, faceIndex: number): Promise<Buffer>;
};

/** `Box` is relative 0-1, but the unclipped detector math can produce values
 * slightly outside that range for a face at the image edge. Clamp the box
 * only — never the keypoints or anything feeding the embedding, since that
 * would corrupt alignment. */
function clampBox(box: Box): Box {
    const left = Math.min(1, Math.max(0, box.left));
    const top = Math.min(1, Math.max(0, box.top));
    const right = Math.min(1, Math.max(0, box.left + box.width));
    const bottom = Math.min(1, Math.max(0, box.top + box.height));
    return { left, top, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
}

export async function createMatcher(): Promise<Matcher> {
    assertModelsPresent();
    const providers = ['coreml', 'cpu'];
    const det = await ort.InferenceSession.create(DET_MODEL, { executionProviders: providers });
    const rec = await ort.InferenceSession.create(REC_MODEL, { executionProviders: providers });

    // The model is the bottleneck; serialize so concurrent requests queue
    // rather than contend. `run` is what callers await (and can reject on),
    // while `queue` swallows the rejection so a failed call doesn't poison
    // the chain for whoever queues up next.
    let queue: Promise<unknown> = Promise.resolve();
    function serialize<T>(fn: () => Promise<T>): Promise<T> {
        const run = queue.then(fn, fn);
        queue = run.catch(() => {});
        return run;
    }

    async function detectFaces(img: RawImage): Promise<DetectedFace[]> {
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
            for (let i = 0; i < detected.length; i++) {
                const f = detected[i];
                faces.push({
                    detScore: f.detScore,
                    box: clampBox(toRelativeBox(f.bbox, img.width, img.height)),
                    embedding: await embedCrop(img, f.kps),
                    faceIndex: i,
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

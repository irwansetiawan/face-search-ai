import test from 'node:test';
import assert from 'node:assert/strict';
import { umeyama, cosine, toRelativeBox, letterbox, decodeDetections, alignCrop, RawImage } from './pipeline.js';

// Canonical ArcFace 5-point reference for a 112x112 crop, inlined from the
// module (not exported — alignCrop's identity-transform test needs it).
const ARCFACE_DST = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

function solidImage(width: number, height: number, gray: number): RawImage {
    const data = Buffer.alloc(width * height * 3, gray);
    return { data, width, height };
}

function patternImage(width: number, height: number): RawImage {
    const data = Buffer.alloc(width * height * 3);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const p = (y * width + x) * 3;
            data[p] = (x * 3 + y * 5) % 256;
            data[p + 1] = (x * 7 + y * 2) % 256;
            data[p + 2] = (x * 11 + y) % 256;
        }
    }
    return { data, width, height };
}

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

// ---------------------------------------------------------------- letterbox

test('letterbox normalizes a fully-covering square image with no padding', () => {
    const img = solidImage(64, 64, 180);
    const { input, scale } = letterbox(img, 32);
    assert.equal(scale, 0.5);
    const expected = (180 - 127.5) / 128.0;
    for (let i = 0; i < input.length; i++) {
        assert.ok(Math.abs(input[i] - expected) < 1e-6, `index ${i} was ${input[i]}`);
    }
});

test('letterbox pads a portrait image on the right with the uint8-zero fill value', () => {
    // height > width -> imRatio > 1 -> nh = size, nw = trunc(size / imRatio).
    const img = solidImage(50, 100, 200);
    const size = 40;
    const { input, scale } = letterbox(img, size);
    assert.equal(scale, size / 100); // det_scale is new_height/orig_height, not a min-ratio

    const nw = Math.trunc(size / (100 / 50)); // = 20
    const plane = size * size;
    const contentVal = (200 - 127.5) / 128.0;
    const fillVal = -127.5 / 128.0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const d = y * size + x;
            const expected = x < nw ? contentVal : fillVal;
            for (let c = 0; c < 3; c++) {
                const v = input[c * plane + d];
                assert.ok(Math.abs(v - expected) < 1e-6,
                    `(${x},${y}) c=${c} was ${v}, expected ${expected}`);
            }
        }
    }
});

test('letterbox truncates target dimensions rather than rounding, for both orientations', () => {
    // Portrait: width=100, height=137 -> imRatio=1.37 -> nh=50, nw=trunc(50/1.37)=36.
    const portrait = solidImage(100, 137, 100);
    const { scale: pScale } = letterbox(portrait, 50);
    assert.equal(pScale, 50 / 137);

    // Landscape: width=200, height=100 -> imRatio=0.5 -> nw=40, nh=trunc(40*0.5)=20.
    const landscape = solidImage(200, 100, 100);
    const size = 40;
    const { input, scale: lScale } = letterbox(landscape, size);
    assert.equal(lScale, 20 / 100);
    const nh = Math.trunc(size * 0.5); // = 20
    const contentVal = (100 - 127.5) / 128.0;
    const fillVal = -127.5 / 128.0;
    // Row just inside content vs. row just inside the bottom padding.
    assert.ok(Math.abs(input[(nh - 1) * size] - contentVal) < 1e-6);
    assert.ok(Math.abs(input[nh * size] - fillVal) < 1e-6);
});

// ---------------------------------------------------------------- decodeDetections

test('decodeDetections decodes, filters by score, and applies NMS', () => {
    const STRIDES = [8, 16, 32];
    const NUM_ANCHORS = 2;
    const DET_SIZE = 640;

    const dims = STRIDES.map(s => Math.ceil(DET_SIZE / s)); // [80, 40, 20]
    const scoresArrs = dims.map(d => new Float32Array(d * d * NUM_ANCHORS));
    const bboxesArrs = dims.map(d => new Float32Array(d * d * NUM_ANCHORS * 4));
    const kpsArrs = dims.map(d => new Float32Array(d * d * NUM_ANCHORS * 10));

    const gw8 = dims[0]; // 80

    // Candidate A: stride-8, grid (x=7,y=5), anchor 0 -> flat index 814.
    const iA = (5 * gw8 + 7) * NUM_ANCHORS + 0;
    scoresArrs[0][iA] = 0.9;
    bboxesArrs[0].set([1.0, 1.5, 2.0, 2.5], iA * 4);
    kpsArrs[0].set([0.1, 0.2, 0.3, -0.1, 0, 0, -0.2, 0.15, 0.25, -0.25], iA * 10);

    // Candidate B: same grid cell, anchor 1 -> identical box, lower score -> NMS-suppressed by A.
    const iB = (5 * gw8 + 7) * NUM_ANCHORS + 1;
    scoresArrs[0][iB] = 0.8;
    bboxesArrs[0].set([1.0, 1.5, 2.0, 2.5], iB * 4);

    // Candidate C: distant grid cell, highest score -> survives, non-overlapping with A.
    const iC = (50 * gw8 + 60) * NUM_ANCHORS + 0;
    scoresArrs[0][iC] = 0.95;
    bboxesArrs[0].set([1.0, 1.0, 1.0, 1.0], iC * 4);

    // Candidate D: below the 0.5 detection floor -> dropped before NMS ever sees it.
    const iD = (10 * gw8 + 10) * NUM_ANCHORS + 0;
    scoresArrs[0][iD] = 0.49;

    const outputs = [...scoresArrs, ...bboxesArrs, ...kpsArrs];
    const faces = decodeDetections(outputs, 1);

    assert.equal(faces.length, 2, 'expected candidate B suppressed by NMS and D dropped by the score floor');

    // Sorted by descending score: C first, then A.
    assert.ok(Math.abs(faces[0].detScore - 0.95) < 1e-6);
    assert.deepEqual(faces[0].bbox, [480 - 8, 400 - 8, 480 + 8, 400 + 8]);

    assert.ok(Math.abs(faces[1].detScore - 0.9) < 1e-6);
    const cx = 7 * 8, cy = 5 * 8;
    const expectedBbox = [cx - 8, cy - 12, cx + 16, cy + 20];
    assert.deepEqual(faces[1].bbox, expectedBbox);
    assert.equal(faces[1].widthPx, expectedBbox[2] - expectedBbox[0]);
    assert.equal(faces[1].heightPx, expectedBbox[3] - expectedBbox[1]);
    const expectedKps = [
        [cx + 0.1 * 8, cy + 0.2 * 8],
        [cx + 0.3 * 8, cy - 0.1 * 8],
        [cx + 0 * 8, cy + 0 * 8],
        [cx - 0.2 * 8, cy + 0.15 * 8],
        [cx + 0.25 * 8, cy - 0.25 * 8],
    ];
    for (let k = 0; k < 5; k++) {
        assert.ok(Math.abs(faces[1].kps[k][0] - expectedKps[k][0]) < 1e-4);
        assert.ok(Math.abs(faces[1].kps[k][1] - expectedKps[k][1]) < 1e-4);
    }
    assert.ok(!faces.some(f => Math.abs(f.detScore - 0.49) < 1e-6), 'sub-threshold candidate must not appear');
});

// ---------------------------------------------------------------- alignCrop

test('alignCrop is the identity sampling + /127.5 normalization when kps equal the ArcFace reference', () => {
    // src === dst -> umeyama returns the identity transform, so alignCrop degenerates
    // to a direct per-pixel remap: out[x,y] = (src[x,y] - 127.5) / 127.5.
    const img = patternImage(120, 120);
    const out = alignCrop(img, ARCFACE_DST, 112);
    const plane = 112 * 112;
    for (let y = 0; y < 112; y++) {
        for (let x = 0; x < 112; x++) {
            const d = y * 112 + x;
            const p = (y * img.width + x) * 3;
            for (let c = 0; c < 3; c++) {
                const expected = (img.data[p + c] - 127.5) / 127.5;
                const actual = out[c * plane + d];
                assert.ok(Math.abs(actual - expected) < 1e-4,
                    `(${x},${y}) c=${c} was ${actual}, expected ${expected}`);
            }
        }
    }
});

test('alignCrop marks out-of-bounds samples as -1 rather than reading past the source', () => {
    // A 112x112 source with an identity transform: the last row/column sample at
    // (111,111) needs source pixel (112,112), which is out of bounds.
    const img = patternImage(112, 112);
    const out = alignCrop(img, ARCFACE_DST, 112);
    const plane = 112 * 112;
    const d = 111 * 112 + 111;
    assert.equal(out[d], -1);
    assert.equal(out[plane + d], -1);
    assert.equal(out[2 * plane + d], -1);

    // An interior pixel, unaffected by the boundary guard, still matches the formula.
    const dInterior = 50 * 112 + 60;
    const pInterior = (50 * img.width + 60) * 3;
    for (let c = 0; c < 3; c++) {
        const expected = (img.data[pInterior + c] - 127.5) / 127.5;
        assert.ok(Math.abs(out[c * plane + dInterior] - expected) < 1e-4);
    }
});

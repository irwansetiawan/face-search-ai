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

test('faceIndex is detection-order and threaded through to alignedCropPng', async () => {
    // g8_group.jpg's largest face lands at a non-zero detection index (verified
    // empirically), which is exactly what makes this fixture useful: it can
    // distinguish "faceIndex correctly selected" from "always face 0".
    //
    // Note: re-embedding an alignedCropPng output via embedLargest is NOT a
    // viable way to check identity here. ArcFace-aligned 112x112 crops are
    // cropped tight to the interior landmarks (no forehead/chin margin), which
    // the SCRFD detector reliably fails to find a face in at all — confirmed
    // even against a large, sharp source face. So identity is checked here via
    // faceIndex's threading through the API, not via re-detection.
    const matcher = await createMatcher();
    const buf = read('g8_group.jpg');
    const { faces } = await matcher.embedAll(buf);
    assert.ok(faces.length >= 2, 'need at least two faces to distinguish faceIndex');

    // The largest face by box area, same selection rule as embedLargest.
    const largest = faces.reduce((a, b) =>
        a.box.width * a.box.height >= b.box.width * b.box.height ? a : b);
    assert.ok(Number.isInteger(largest.faceIndex));
    assert.ok(largest.faceIndex >= 0 && largest.faceIndex < faces.length);

    const viaLargest = await matcher.embedLargest(buf);
    assert.equal(viaLargest!.faceIndex, largest.faceIndex,
        'embedLargest must report the same detection-order index as embedAll\'s largest face');

    // faceIndex must actually be threaded through to alignedCropPng, not
    // ignored in favor of always cropping face 0 — a bug that would produce a
    // saved person's avatar showing a different person's face while the
    // embedding is correct (Ruling 2's failure mode).
    assert.notEqual(largest.faceIndex, 0,
        'fixture assumption: the largest face must not be detection index 0 for this check to catch a hardcoded-0 bug');
    const cropLargest = await matcher.alignedCropPng(buf, largest.faceIndex);
    const cropFirst = await matcher.alignedCropPng(buf, 0);
    assert.ok(!cropLargest.equals(cropFirst),
        'alignedCropPng(faceIndex) must vary with faceIndex, not always crop the same face');
});

test('a rejected alignedCropPng call does not poison the queue for later callers', async () => {
    const matcher = await createMatcher();
    const buf = read('obama_portrait.jpg');

    await assert.rejects(() => matcher.alignedCropPng(buf, 999));

    // A subsequent call on the same matcher must still succeed.
    const face = await matcher.embedLargest(buf);
    assert.ok(face, 'matcher must remain usable after a rejected call');
});

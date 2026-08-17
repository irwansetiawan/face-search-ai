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
const face = (embedding: Float32Array): EmbeddedFace =>
    ({ detScore: 0.9, box, embedding, faceIndex: 0 });

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

test('picks the correct best reference per face independently', () => {
    // Two references far apart; two faces, each closest to a different one.
    // Give them distinct similarity magnitudes (0.9 vs 0.8) so the descending
    // sort order is unambiguous and the test doesn't depend on sort stability.
    const refA = unit(1, 0), refB = unit(0, 1);
    const faceCloseToA = face(unit(0.9, Math.sqrt(1 - 0.9 * 0.9)));
    const faceCloseToB = face(unit(Math.sqrt(1 - 0.8 * 0.8), 0.8));
    const [scoredA, scoredB] = scoreFaces([faceCloseToA, faceCloseToB], [refA, refB], 0.4);
    assert.equal(scoredA.bestReference, 0);
    assert.ok(Math.abs(scoredA.cosine - 0.9) < 1e-6);
    assert.equal(scoredB.bestReference, 1);
    assert.ok(Math.abs(scoredB.cosine - 0.8) < 1e-6);
});

test('threshold is inclusive: a face scoring exactly at threshold is matched', () => {
    const ref = unit(1, 0);
    // Construct an embedding whose cosine similarity to ref is exactly 0.4.
    const other = unit(0, 1);
    const combo = new Float32Array(512);
    for (let i = 0; i < 512; i++) combo[i] = 0.4 * ref[i] + Math.sqrt(1 - 0.4 * 0.4) * other[i];
    const [scored] = scoreFaces([face(combo)], [ref], 0.4);
    assert.ok(Math.abs(scored.cosine - 0.4) < 1e-6, `expected cosine ~0.4, got ${scored.cosine}`);
    assert.equal(scored.matched, true);
});

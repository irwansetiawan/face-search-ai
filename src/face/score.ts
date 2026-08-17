import { cosine, type Box } from './pipeline.js';
import type { EmbeddedFace } from './matcher.js';

const DEFAULT_MATCH_THRESHOLD = 0.4;

/**
 * Resolves FACE_MATCH_THRESHOLD from a raw env value. `??` alone is not
 * enough here: it substitutes only on `undefined`, so two misconfigurations
 * would otherwise pass through silently --
 *   - `FACE_MATCH_THRESHOLD=` (empty/whitespace string): `Number('')` is
 *     `0`, a threshold that matches every face in every photo.
 *   - a non-numeric value: `Number('abc')` is `NaN`, which matches nothing
 *     (every comparison against NaN is false).
 * Both are silent, total behavioral failures -- the server's boot log would
 * print the bad value as though it were a valid threshold. Anything that
 * isn't a finite number after trimming falls back to the default and warns
 * loudly on stderr instead.
 */
export function resolveThreshold(raw: string | undefined): number {
    const trimmed = raw?.trim();
    if (!trimmed) {
        if (raw !== undefined) {
            console.warn(
                `FACE_MATCH_THRESHOLD is set but empty/whitespace; ` +
                `falling back to the default ${DEFAULT_MATCH_THRESHOLD}.`
            );
        }
        return DEFAULT_MATCH_THRESHOLD;
    }
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) {
        console.warn(
            `FACE_MATCH_THRESHOLD="${raw}" is not a valid number; ` +
            `falling back to the default ${DEFAULT_MATCH_THRESHOLD}.`
        );
        return DEFAULT_MATCH_THRESHOLD;
    }
    return parsed;
}

// Provisional operating point, not a calibrated one. It has not been tuned
// against a labelled dataset -- it is a starting value to unblock the rest
// of the pipeline, and is expected to move once real precision/recall data
// exists.
export const MATCH_THRESHOLD = resolveThreshold(process.env.FACE_MATCH_THRESHOLD);

export type ScoredFace = {
    box: Box;
    cosine: number;
    detScore: number;
    matched: boolean;
    bestReference: number;
};

/**
 * Score each detected face against every reference embedding, keeping the
 * best (max-over-references). A person may have several reference photos; a
 * candidate face only needs to resemble one of them. That is what absorbs
 * pose and lighting variation -- a profile shot and a front-on shot sit far
 * apart in embedding space, so requiring a match against every reference
 * (or an average) would reject faces that a single good reference would
 * catch.
 *
 * Both `f.embedding` and each `ref` arrive already L2-normalized (Task 3 /
 * the `/probe` route), so `cosine` -- a bare dot product -- IS cosine
 * similarity here without renormalizing.
 */
export function scoreFaces(
    faces: EmbeddedFace[], refs: Float32Array[], threshold: number
): ScoredFace[] {
    return faces
        .map(f => {
            let best = -Infinity;
            let bestReference = -1;
            refs.forEach((ref, i) => {
                const sim = cosine(f.embedding, ref);
                if (sim > best) {
                    best = sim;
                    bestReference = i;
                }
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

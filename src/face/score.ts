import { cosine, type Box } from './pipeline.js';
import type { EmbeddedFace } from './matcher.js';

// Provisional operating point, not a calibrated one. It has not been tuned
// against a labelled dataset -- it is a starting value to unblock the rest
// of the pipeline, and is expected to move once real precision/recall data
// exists.
export const MATCH_THRESHOLD = Number(process.env.FACE_MATCH_THRESHOLD ?? 0.4);

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

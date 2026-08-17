/**
 * Recomputes stored embeddings when the face pipeline has changed since a
 * store was last written. Stored references keep a downscaled copy of the
 * original photo specifically so this is possible without asking the user
 * to re-upload every reference photo.
 *
 * Runs once at boot (see Task 10's server.ts wiring), before the server
 * starts accepting requests. `list()`/`get()` return live references into
 * the store's internal state, so relying on `store.updateReferenceEmbedding`
 * to mutate a `Person`'s reference in place (visible through the same object
 * this module already holds) is safe here only because nothing else can be
 * mutating the store concurrently — this module does not add any
 * concurrency protection of its own.
 */
import fs from 'fs/promises';
import path from 'path';
import type { Store, Person } from './store.js';
import { PIPELINE_VERSION } from './store.js';
import type { Matcher } from '../face/matcher.js';

/**
 * Recompute every stored embedding from its original image when
 * `storedVersion` does not match the code's current `PIPELINE_VERSION`.
 * Returns the number of references actually re-embedded.
 *
 * A reference whose image file is missing or unreadable, or in which no face
 * is found, keeps its stale embedding rather than being dropped: losing
 * someone's saved person because one file went missing would be worse than
 * serving a slightly outdated vector for it. Each skip is logged
 * individually (who, which reference, which file, why), plus one summary
 * line if anything was skipped, so it's visible in boot output rather than
 * invisible.
 *
 * Every successfully re-embedded reference is persisted immediately via
 * `store.updateReferenceEmbedding`, which does NOT stamp `pipelineVersion`.
 * The store's on-disk version is only ever advanced to `PIPELINE_VERSION`
 * (via `replaceAll`, at the very end) when `skipped === 0` -- i.e. every
 * single reference in the store re-embedded cleanly this run. That is what
 * makes the retry guarantee actually hold: if even one reference is skipped,
 * the file stays observably stale, so THIS ENTIRE RUN is retried on the next
 * boot -- not just the reference that failed. A store with one permanently
 * broken reference (file gone for good, never fixed) therefore re-embeds
 * every OTHER reference on every future boot too -- a deliberate trade:
 * retrying forever is the cost of never silently letting a fixable stale
 * reference stop retrying.
 */
export async function reembedIfStale(
    store: Store, matcher: Matcher, storedVersion: number
): Promise<number> {
    if (storedVersion === PIPELINE_VERSION) return 0;

    let updated = 0;
    let skipped = 0;
    const people: Person[] = store.list();
    for (const person of people) {
        for (const ref of person.references) {
            const abs = path.join(store.dataDir, ref.image);
            try {
                const bytes = await fs.readFile(abs);
                const face = await matcher.embedLargest(bytes);
                if (!face) {
                    skipped++;
                    console.warn(
                        `reembed: no face found for person ${person.id} reference ${ref.id} (${abs}); keeping stale embedding`
                    );
                    continue;
                }
                // Persisted immediately and individually -- not batched into
                // a single end-of-run write -- so a successful re-embed
                // survives on disk even if a later reference in this same
                // run fails and the run as a whole never reaches the
                // clean-sweep replaceAll below.
                await store.updateReferenceEmbedding(
                    person.id, ref.id, Array.from(face.embedding), face.detScore);
                updated++;
            } catch (err) {
                skipped++;
                console.warn(
                    `reembed: failed to re-embed person ${person.id} reference ${ref.id} (${abs}); keeping stale embedding`,
                    err
                );
            }
        }
    }
    if (skipped > 0) {
        console.warn(`reembed: ${updated} reference(s) re-embedded, ${skipped} skipped and left stale`);
    }
    // Only a fully clean sweep may advance the version stamp -- see the
    // doc comment above for why a partial success must not.
    if (updated > 0 && skipped === 0) await store.replaceAll(people);
    return updated;
}

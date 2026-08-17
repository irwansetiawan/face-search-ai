/**
 * Recomputes stored embeddings when the face pipeline has changed since a
 * store was last written. Stored references keep a downscaled copy of the
 * original photo specifically so this is possible without asking the user
 * to re-upload every reference photo.
 *
 * Runs once at boot (see Task 10's server.ts wiring), before the server
 * starts accepting requests. `list()`/`get()` return live references into
 * the store's internal state, so mutating the `Person` objects in place and
 * then persisting with `replaceAll` is safe here only because nothing else
 * can be mutating the store concurrently — this module does not add any
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
 * A reference whose image file is missing or unreadable keeps its stale
 * embedding rather than being dropped: losing someone's saved person because
 * one file went missing would be worse than serving a slightly outdated
 * vector for it.
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
                const bytes = await fs.readFile(abs);
                const face = await matcher.embedLargest(bytes);
                if (!face) continue;
                ref.embedding = Array.from(face.embedding);
                ref.detScore = face.detScore;
                updated++;
            } catch {
                // Missing or unreadable image: keep the stale vector and
                // move on rather than dropping the reference.
            }
        }
    }
    if (updated > 0) await store.replaceAll(people);
    return updated;
}

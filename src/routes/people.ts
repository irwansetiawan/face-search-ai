/**
 * HTTP surface over the saved-people store (Task 8). A saved person is a
 * name plus one or more reference photos; for each reference this module
 * stores three artefacts: a downscaled original (so Task 9's
 * `reembedIfStale` can re-derive the embedding after a pipeline change
 * without asking the user to re-upload), an aligned 112x112 crop used as the
 * picker avatar, and the 512-float embedding itself (kept server-side only —
 * see `publicPerson`).
 */
import express from 'express';
import multer from 'multer';
import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';
import sharp from 'sharp';
import type { Store, NewReference, Person } from '../people/store.js';
import type { Matcher, EmbeddedFace } from '../face/matcher.js';

// Detection runs at 640px; a stored original larger than this is disk space
// a re-embed could never make use of.
const MAX_STORED_EDGE = 1600;

/** Public shape for the people list/picker: never ship the 512-float
 * embedding to the browser -- it serves no purpose there and would bloat
 * every response for no benefit. */
function publicPerson(p: Person) {
    return {
        id: p.id,
        name: p.name,
        createdAt: p.createdAt,
        references: p.references.map(r => ({
            id: r.id, thumb: r.thumb, detScore: r.detScore, addedAt: r.addedAt,
        })),
    };
}

export function peopleRouter(store: Store, matcher: Matcher): express.Router {
    const router = express.Router();
    const upload = multer({ dest: 'uploads/' });

    /**
     * Writes both stored artefacts (downscaled original + aligned avatar
     * crop, cropped from `face.faceIndex` -- NOT a hardcoded 0, since
     * embedLargest ranks by clamped box area and is not necessarily the
     * first face detected; passing 0 would crop whichever face happened to
     * be detected first, a different face from the one the embedding
     * actually came from on any multi-face photo where the largest face
     * isn't first) under `people/<dirId>/`.
     *
     * Both files are written to their FINAL resting path before this
     * function returns, and only once it returns does a caller ever call
     * `store.addPerson`/`addReference` with those paths -- which is what
     * upholds the ordering invariant: the store must never persist a path
     * to a file that isn't already there.
     *
     * Everything this function can throw (disk full, permissions, a bug in
     * `alignedCropPng`) is a server-side fault, not the client's -- `face`
     * has already been successfully decoded from the client's bytes by the
     * time this is called. Callers must NOT fold a throw from here into
     * `unreadable_image`.
     */
    async function writeArtifacts(dirId: string, buffer: Buffer, face: EmbeddedFace): Promise<NewReference> {
        const dir = path.join(store.dataDir, 'people', dirId);
        await fs.mkdir(dir, { recursive: true });
        const stamp = `${Date.now().toString(36)}${crypto.randomBytes(3).toString('hex')}`;

        const imageRel = path.join('people', dirId, `${stamp}.jpg`);
        const thumbRel = path.join('people', dirId, `${stamp}.thumb.png`);
        try {
            await sharp(buffer)
                .rotate() // bake in EXIF orientation before storing
                .resize(MAX_STORED_EDGE, MAX_STORED_EDGE, { fit: 'inside', withoutEnlargement: true })
                .jpeg({ quality: 88 })
                .toFile(path.join(store.dataDir, imageRel));

            await fs.writeFile(
                path.join(store.dataDir, thumbRel),
                await matcher.alignedCropPng(buffer, face.faceIndex),
            );
        } catch (err) {
            // Don't leave an orphaned .jpg behind if the thumb write (or
            // alignedCropPng) fails after the downscaled original already
            // landed on disk -- for addReference this directory belongs to
            // an existing, already-registered person, so nothing here is
            // cleaned up automatically by a caller-side rollback.
            await fs.unlink(path.join(store.dataDir, imageRel)).catch(() => {});
            await fs.unlink(path.join(store.dataDir, thumbRel)).catch(() => {});
            throw err;
        }

        return {
            image: imageRel,
            thumb: thumbRel,
            embedding: Array.from(face.embedding),
            detScore: face.detScore,
        };
    }

    router.get('/', (_req, res) => {
        res.json(store.list().map(publicPerson));
    });

    router.post('/', upload.single('photo'), async (req, res) => {
        const file = req.file;
        const name = String(req.body?.name ?? '').trim();
        if (!file || !name) {
            if (file) await fs.unlink(file.path).catch(() => {});
            res.status(400).json({ error: 'name_and_photo_required' });
            return;
        }

        let status = 500;
        let body: Record<string, unknown> = { error: 'create_person_failed' };
        try {
            // multer already wrote this file to disk before the handler ran,
            // so a read failure here is a server-side fault (disk,
            // permissions, environment) -- never the client's.
            let buffer: Buffer | undefined;
            try {
                buffer = await fs.readFile(file.path);
            } catch (err) {
                console.error('people: failed to read uploaded temp file', err);
            }

            if (buffer) {
                // Only this decode step -- matcher.embedLargest's first
                // move is to decode `buffer` as an image (via sharp) -- maps
                // a throw to 400 unreadable_image. Everything after it
                // (writing files, talking to the store) is a server-side
                // fault if it throws, not the client's, and must fall
                // through to the 500 default instead.
                let face: EmbeddedFace | null | undefined;
                try {
                    face = await matcher.embedLargest(buffer);
                } catch (err) {
                    console.error('people: failed to decode uploaded photo', err);
                    status = 400;
                    body = { error: 'unreadable_image' };
                }

                if (face === null) {
                    status = 422;
                    body = { error: 'no_face_detected' };
                } else if (face) {
                    // The real person id only exists once store.addPerson
                    // returns, so writeArtifacts first writes under a
                    // disposable, locally-generated directory name.
                    // addPerson is called with paths that already point at
                    // real files (upholding the ordering invariant); the
                    // artefacts are then COPIED (not renamed) into a
                    // directory named after the real person id, so
                    // store.ts's deletePerson (which assumes people/<id>/)
                    // can clean them up later. Copying instead of renaming
                    // means both locations hold valid files for the entire
                    // window between "final copy exists" and "store points
                    // at it" -- the store's persisted path is never allowed
                    // to go stale, which a rename-then-patch would risk if
                    // the process died between the two steps.
                    const pendingId = `pending_${crypto.randomBytes(6).toString('hex')}`;
                    // Tracks whether store.addPerson actually took ownership
                    // of the pendingId paths (i.e. its flush succeeded and a
                    // persisted record now references them). Declared
                    // outside the try so the catch below can see it: only
                    // once this is true is rm'ing the pending directory on a
                    // later failure forbidden -- before that point it would
                    // just be cleaning up writeArtifacts' own output, which
                    // nothing persisted references yet.
                    let registered = false;
                    try {
                        const ref = await writeArtifacts(pendingId, buffer, face);
                        const person = await store.addPerson(name, ref);
                        registered = true;
                        const first = person.references[0];
                        const finalDir = path.join(store.dataDir, 'people', person.id);
                        const newImageRel = first.image.replace(pendingId, person.id);
                        const newThumbRel = first.thumb.replace(pendingId, person.id);

                        let copied = false;
                        try {
                            await fs.mkdir(finalDir, { recursive: true });
                            await fs.copyFile(
                                path.join(store.dataDir, first.image), path.join(store.dataDir, newImageRel));
                            await fs.copyFile(
                                path.join(store.dataDir, first.thumb), path.join(store.dataDir, newThumbRel));
                            copied = true;
                        } catch (err) {
                            // Nothing has been persisted or mutated yet at
                            // this point, so the person's already-flushed
                            // pending-path references are still exactly
                            // correct. Degraded naming only (deletePerson's
                            // cleanup won't find this dir later), not a
                            // broken reference -- not surfaced as a request
                            // failure.
                            console.error(
                                `people: failed to copy artefacts from ${pendingId} to ${person.id}; ` +
                                'keeping the pending directory name', err,
                            );
                            await fs.rm(finalDir, { recursive: true, force: true }).catch(() => {});
                        }

                        if (copied) {
                            // Both artefacts now exist at BOTH the pending
                            // and final paths. updateReferencePaths mutates
                            // and flushes inside the store's own serialized
                            // queue and does NOT stamp pipelineVersion --
                            // unlike replaceAll(store.list()), which both
                            // races a concurrent mutation captured outside
                            // that queue and would incorrectly mark the
                            // whole store "current". If this throws, the
                            // pending-path references (already flushed) are
                            // still valid on disk (nothing was removed
                            // yet), so nothing breaks; the final copies are
                            // just left behind as a harmless duplicate.
                            try {
                                await store.updateReferencePaths(person.id, first.id, newImageRel, newThumbRel);
                                await fs.rm(path.join(store.dataDir, 'people', pendingId),
                                    { recursive: true, force: true }).catch(() => {});
                            } catch (err) {
                                console.error(
                                    `people: failed to persist relabelled paths for ${person.id}; ` +
                                    'both copies remain on disk', err,
                                );
                            }
                        }

                        status = 201;
                        body = publicPerson(person);
                    } catch (err) {
                        console.error('people: failed to create person', err);
                        // status/body remain the 500 default -- writing
                        // artefacts or talking to the store failing is a
                        // server-side fault, not the client's. If addPerson
                        // never resolved (registered stays false -- e.g. its
                        // flush() hit a disk/permissions fault), nothing
                        // persisted references the pending directory yet, so
                        // clean it up rather than leaving it orphaned. Once
                        // registered is true, a persisted record DOES
                        // reference those paths (or, further down, the
                        // relabelled ones) -- rm'ing here would break the
                        // exact invariant this route exists to uphold.
                        if (!registered) {
                            await fs.rm(path.join(store.dataDir, 'people', pendingId),
                                { recursive: true, force: true }).catch(() => {});
                        }
                    }
                }
            }
        } finally {
            await fs.unlink(file.path).catch(() => {});
        }

        res.status(status).json(body);
    });

    router.post('/:id/references', upload.single('photo'), async (req, res) => {
        const file = req.file;
        if (!file) {
            res.status(400).json({ error: 'photo_required' });
            return;
        }

        let status = 500;
        let body: Record<string, unknown> = { error: 'add_reference_failed' };
        try {
            if (!store.get(req.params.id)) {
                status = 404;
                body = { error: 'no_such_person' };
            } else {
                let buffer: Buffer | undefined;
                try {
                    buffer = await fs.readFile(file.path);
                } catch (err) {
                    console.error('people: failed to read uploaded temp file', err);
                }

                if (buffer) {
                    let face: EmbeddedFace | null | undefined;
                    try {
                        face = await matcher.embedLargest(buffer);
                    } catch (err) {
                        console.error('people: failed to decode uploaded photo', err);
                        status = 400;
                        body = { error: 'unreadable_image' };
                    }

                    if (face === null) {
                        status = 422;
                        body = { error: 'no_face_detected' };
                    } else if (face) {
                        try {
                            // The person already exists, so its final
                            // directory name is already known -- no
                            // pending/copy dance needed here, unlike person
                            // creation.
                            const ref = await writeArtifacts(req.params.id, buffer, face);
                            const person = await store.addReference(req.params.id, ref);
                            status = 200;
                            body = publicPerson(person);
                        } catch (err) {
                            console.error('people: failed to add reference', err);
                            // status/body remain the 500 default.
                        }
                    }
                }
            }
        } finally {
            await fs.unlink(file.path).catch(() => {});
        }

        res.status(status).json(body);
    });

    router.delete('/:id', async (req, res) => {
        let status = 500;
        let body: Record<string, unknown> | null = { error: 'delete_person_failed' };
        try {
            // The store's deletePerson is a silent no-op on an unknown id
            // (deliberate, per store.ts); this route checks existence itself
            // so a nonexistent person 404s instead of reporting success.
            if (!store.get(req.params.id)) {
                status = 404;
                body = { error: 'no_such_person' };
            } else {
                await store.deletePerson(req.params.id);
                status = 204;
                body = null;
            }
        } catch (err) {
            console.error('people: failed to delete person', err);
            status = 500;
            body = { error: 'delete_person_failed' };
        }

        if (body === null) res.status(status).end();
        else res.status(status).json(body);
    });

    router.delete('/:id/references/:refId', async (req, res) => {
        let status = 500;
        let body: Record<string, unknown> = { error: 'delete_reference_failed' };
        try {
            const person = store.get(req.params.id);
            if (!person) {
                status = 404;
                body = { error: 'no_such_person' };
            } else if (!person.references.some(r => r.id === req.params.refId)) {
                status = 404;
                body = { error: 'no_such_reference' };
            } else if (person.references.length <= 1) {
                // Checked here, not just left to the store's thrown error,
                // so a genuine unexpected failure from deleteReference below
                // (e.g. a disk error during flush) maps to 500 rather than
                // being swallowed into the same 409 as this expected refusal.
                status = 409;
                body = { error: 'cannot_delete_last_reference' };
            } else {
                const updated = await store.deleteReference(req.params.id, req.params.refId);
                status = 200;
                body = publicPerson(updated);
            }
        } catch (err) {
            console.error('people: failed to delete reference', err);
            status = 500;
            body = { error: 'delete_reference_failed' };
        }

        res.status(status).json(body);
    });

    return router;
}

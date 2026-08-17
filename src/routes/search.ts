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

/**
 * `/probe` (Task 4) embeds a source face once into 512 floats; `/search` is
 * called once per target photo and scores every face in it against those
 * floats. This split is the entire point of the migration -- the old
 * Rekognition-backed code re-uploaded and re-detected the source image for
 * every photo scanned.
 *
 * A target photo with no faces is a routine outcome for a folder scan
 * ("nobody in this photo"), not an error -- unlike Rekognition, which raised
 * an exception. That case is a 200 with `faces: []` here.
 */
export function searchHandler(matcher: Matcher): RequestHandler {
    return async (req, res) => {
        const files = req.files as { [f: string]: Express.Multer.File[] };
        const upload = files?.target?.[0];
        if (!upload) {
            res.status(400).json({ error: 'unreadable_image' });
            return;
        }

        // Same shape as probeHandler (Task 4): compute the outcome, clean up
        // the temp file, and only then send the response. Sending inside the
        // try and unlinking in `finally` would let a client observe a
        // response while the temp file still exists on disk.
        let status = 500;
        let body: Record<string, unknown> = { error: 'search_failed' };
        try {
            const refs = parseProbe(req.body?.probe);
            if (!refs) {
                status = 400;
                body = { error: 'bad_probe' };
            } else {
                // multer already wrote this file to disk before the handler
                // ran, so a read failure here is a server-side fault (disk,
                // permissions, environment) -- never the client's.
                let buffer: Buffer | undefined;
                try {
                    buffer = await fs.readFile(upload.path);
                } catch (err) {
                    console.error('search: failed to read uploaded temp file', err);
                }

                if (buffer) {
                    try {
                        const { faces } = await matcher.embedAll(buffer);
                        const scored = scoreFaces(faces, refs, MATCH_THRESHOLD);
                        status = 200;
                        body = {
                            threshold: MATCH_THRESHOLD,
                            matched: scored.some(f => f.matched),
                            faces: scored,
                        };
                    } catch (err) {
                        // embedAll's first step decodes the uploaded bytes as
                        // an image (via sharp). A thrown error here means the
                        // bytes are not a decodable image -- the client's
                        // fault, per the documented contract.
                        console.error('search: failed to decode target image', err);
                        status = 400;
                        body = { error: 'unreadable_image' };
                    }
                }
            }
        } finally {
            await fs.unlink(upload.path).catch(() => {});
        }

        res.status(status).json(body);
    };
}

import type { RequestHandler } from 'express';
import fs from 'fs/promises';
import type { Matcher } from '../face/matcher.js';

/**
 * Embeds the source face once into a 512-d vector, so `/search` (Task 5) can
 * score every target photo against these numbers instead of re-uploading and
 * re-detecting the source face per photo.
 */
export function probeHandler(matcher: Matcher): RequestHandler {
    return async (req, res) => {
        const files = req.files as { [f: string]: Express.Multer.File[] };
        const upload = files?.source?.[0];
        if (!upload) {
            res.status(400).json({ error: 'bad_probe' });
            return;
        }

        // The outcome is computed here and the temp file is unlinked in
        // `finally`, both *before* the response is sent below. That ordering
        // matters: it means a client can never observe a response while the
        // temp file still exists on disk, closing the read-after-respond
        // race that a "send inside try, unlink in finally" shape would leave
        // open.
        let status = 500;
        let body: Record<string, unknown> = { error: 'probe_failed' };
        try {
            // multer already wrote this file to disk before the handler ran,
            // so a read failure here is a server-side fault (disk,
            // permissions, environment) -- never the client's. It is
            // reported as an opaque 500 and logged, not folded into
            // unreadable_image. status/body keep their 500 default.
            let buffer: Buffer | undefined;
            try {
                buffer = await fs.readFile(upload.path);
            } catch (err) {
                console.error('probe: failed to read uploaded temp file', err);
            }

            if (buffer) {
                try {
                    const face = await matcher.embedLargest(buffer);
                    if (face) {
                        status = 200;
                        body = {
                            embedding: Array.from(face.embedding),
                            face: { box: face.box, score: face.detScore },
                        };
                    } else {
                        status = 422;
                        body = { error: 'no_face_detected' };
                    }
                } catch {
                    // embedLargest's first step decodes the uploaded bytes
                    // as an image (via sharp). A thrown error here means the
                    // bytes are not a decodable image -- that IS the
                    // client's fault, per the documented contract.
                    status = 400;
                    body = { error: 'unreadable_image' };
                }
            }
        } finally {
            // Always clean up; the old code unlinked only on the success path.
            await fs.unlink(upload.path).catch(() => {});
        }

        res.status(status).json(body);
    };
}

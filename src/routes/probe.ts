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
        try {
            const buffer = await fs.readFile(upload.path);
            const face = await matcher.embedLargest(buffer);
            if (!face) {
                res.status(422).json({ error: 'no_face_detected' });
                return;
            }
            res.status(200).json({
                embedding: Array.from(face.embedding),
                face: { box: face.box, score: face.detScore },
            });
        } catch {
            res.status(400).json({ error: 'unreadable_image' });
        } finally {
            // Always clean up; the old code unlinked only on the success path.
            await fs.unlink(upload.path).catch(() => {});
        }
    };
}

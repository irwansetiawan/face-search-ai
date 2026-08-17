import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import path from 'path';
import { createMatcher, type Matcher } from '../face/matcher.js';
import { createApp } from '../app.js';

const IMAGES = path.join(process.cwd(), 'spike', 'images');
const UPLOADS = path.join(process.cwd(), 'uploads');

// Ruling 14: share one matcher per test file. createMatcher() compiles the
// ONNX graphs (~5s), so creating it per-test would make the suite
// progressively slower across the whole plan. The matcher holds no
// per-request state beyond its serialization queue, so sharing it across
// tests in this file is safe.
let matcher: Matcher | undefined;
async function getMatcher(): Promise<Matcher> {
    return matcher ??= await createMatcher();
}

async function withServer(fn: (base: string) => Promise<void>) {
    const app = createApp(await getMatcher());
    const server = app.listen(0);
    await new Promise(r => server.once('listening', r));
    const { port } = server.address() as { port: number };
    try {
        await fn(`http://127.0.0.1:${port}`);
    } finally {
        await new Promise<void>(resolve => server.close(() => resolve()));
    }
}

function form(field: string, file: string) {
    const fd = new FormData();
    fd.append(field, new Blob([fs.readFileSync(path.join(IMAGES, file))]), file);
    return fd;
}

/** Snapshot of `uploads/` entries, for asserting multer temp files were
 * cleaned up (or not created) around a request. */
function uploadsSnapshot(): Set<string> {
    return new Set(fs.readdirSync(UPLOADS));
}

test('POST /probe returns a 512-d embedding and a relative box, and cleans up the temp file', async () => {
    await withServer(async (base) => {
        const before = uploadsSnapshot();

        const res = await fetch(`${base}/probe`, {
            method: 'POST', body: form('source', 'obama_portrait.jpg'),
        });
        assert.equal(res.status, 200);
        const body = await res.json();
        assert.equal(body.embedding.length, 512);
        assert.ok(body.face.score > 0.5);
        assert.ok(body.face.box.width > 0 && body.face.box.width <= 1);

        assert.deepEqual(uploadsSnapshot(), before,
            'multer temp file must be unlinked on the success path');
    });
});

test('POST /probe returns 422 and cleans up the temp file when there is no face', async () => {
    await withServer(async (base) => {
        const before = uploadsSnapshot();

        const blank = await (await import('sharp')).default({
            create: { width: 200, height: 200, channels: 3, background: '#888' },
        }).jpeg().toBuffer();
        const fd = new FormData();
        fd.append('source', new Blob([blank]), 'blank.jpg');

        const res = await fetch(`${base}/probe`, { method: 'POST', body: fd });
        assert.equal(res.status, 422);
        assert.equal((await res.json()).error, 'no_face_detected');

        assert.deepEqual(uploadsSnapshot(), before,
            'multer temp file must be unlinked on the no-face path');
    });
});

test('POST /probe returns 400 and deletes the multer temp file when the upload is unreadable', async () => {
    await withServer(async (base) => {
        const before = uploadsSnapshot();

        const fd = new FormData();
        fd.append('source', new Blob([Buffer.from('this is not an image')]), 'junk.jpg');
        const res = await fetch(`${base}/probe`, { method: 'POST', body: fd });

        assert.equal(res.status, 400);
        assert.equal((await res.json()).error, 'unreadable_image');
        assert.deepEqual(uploadsSnapshot(), before,
            'multer temp file must be unlinked on the unreadable-image path, not just on success');
    });
});

test('POST /probe returns 400 when no source field is present', async () => {
    await withServer(async (base) => {
        const res = await fetch(`${base}/probe`, { method: 'POST', body: new FormData() });
        assert.equal(res.status, 400);
        assert.equal((await res.json()).error, 'bad_probe');
    });
});

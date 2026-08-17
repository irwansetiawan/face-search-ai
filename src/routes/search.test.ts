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

/** Snapshot of `uploads/` entries, for asserting multer temp files were
 * cleaned up (or not created) around a request. */
function uploadsSnapshot(): Set<string> {
    return new Set(fs.readdirSync(UPLOADS));
}

async function probe(base: string, file: string): Promise<number[]> {
    const fd = new FormData();
    fd.append('source', new Blob([fs.readFileSync(path.join(IMAGES, file))]), file);
    const res = await fetch(`${base}/probe`, { method: 'POST', body: fd });
    return (await res.json()).embedding;
}

function searchForm(embedding: number[], file: string, filename = file) {
    const fd = new FormData();
    fd.append('target', new Blob([fs.readFileSync(path.join(IMAGES, file))]), filename);
    fd.append('probe', JSON.stringify([embedding]));
    return fd;
}

async function search(base: string, embedding: number[], file: string) {
    const res = await fetch(`${base}/search`, { method: 'POST', body: searchForm(embedding, file) });
    return { status: res.status, body: await res.json() };
}

test('finds the probe subject in a crowd and rejects everyone else', async () => {
    await withServer(async (base) => {
        const before = uploadsSnapshot();
        const embedding = await probe(base, 'obama_portrait.jpg');
        const { status, body } = await search(base, embedding, 'obama_cabinet.jpg');

        assert.equal(status, 200);
        assert.equal(body.threshold, 0.4);
        assert.equal(body.matched, true);
        assert.equal(body.faces.filter((f: any) => f.matched).length, 1,
            'exactly one face in the cabinet photo should match');
        assert.ok(body.faces.length >= 20);
        // Multi-reference scoring result shape.
        const top = body.faces[0];
        assert.equal(top.bestReference, 0);
        assert.ok(top.cosine >= body.faces[1].cosine, 'faces must be sorted by descending cosine');

        assert.deepEqual(uploadsSnapshot(), before,
            'multer temp file for the target must be unlinked on the success path');
    });
});

test('reports no match for a photo of someone else', async () => {
    await withServer(async (base) => {
        const embedding = await probe(base, 'obama_portrait.jpg');
        const { status, body } = await search(base, embedding, 'biden_portrait.jpg');
        assert.equal(status, 200);
        assert.equal(body.matched, false);
        assert.ok(body.faces.every((f: any) => !f.matched));
    });
});

test('a photo with no faces is 200 with an empty list, not an error', async () => {
    await withServer(async (base) => {
        const before = uploadsSnapshot();
        const embedding = await probe(base, 'obama_portrait.jpg');
        const blank = await (await import('sharp')).default({
            create: { width: 200, height: 200, channels: 3, background: '#888' },
        }).jpeg().toBuffer();
        const fd = new FormData();
        fd.append('target', new Blob([blank]), 'blank.jpg');
        fd.append('probe', JSON.stringify([embedding]));

        const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
        assert.equal(res.status, 200);
        const body = await res.json();
        assert.deepEqual(body.faces, []);
        assert.equal(body.matched, false);

        assert.deepEqual(uploadsSnapshot(), before,
            'multer temp file must be unlinked on the no-faces path');
    });
});

test('a malformed probe (wrong embedding length) is rejected with 400 and cleans up the temp file', async () => {
    await withServer(async (base) => {
        const before = uploadsSnapshot();
        const fd = new FormData();
        fd.append('target', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_alt.jpg'))]), 'x.jpg');
        fd.append('probe', JSON.stringify([[1, 2, 3]]));  // wrong length
        const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
        assert.equal(res.status, 400);
        assert.equal((await res.json()).error, 'bad_probe');

        assert.deepEqual(uploadsSnapshot(), before,
            'multer temp file must be unlinked on the bad_probe path');
    });
});

test('probe field missing entirely is rejected with 400 bad_probe', async () => {
    await withServer(async (base) => {
        const fd = new FormData();
        fd.append('target', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_alt.jpg'))]), 'x.jpg');
        const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
        assert.equal(res.status, 400);
        assert.equal((await res.json()).error, 'bad_probe');
    });
});

test('probe that is not valid JSON is rejected with 400 bad_probe', async () => {
    await withServer(async (base) => {
        const fd = new FormData();
        fd.append('target', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_alt.jpg'))]), 'x.jpg');
        fd.append('probe', 'not json');
        const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
        assert.equal(res.status, 400);
        assert.equal((await res.json()).error, 'bad_probe');
    });
});

test('an unreadable target image is rejected with 400 unreadable_image and cleans up the temp file', async () => {
    await withServer(async (base) => {
        const before = uploadsSnapshot();
        const embedding = await probe(base, 'obama_portrait.jpg');
        const fd = new FormData();
        fd.append('target', new Blob([Buffer.from('this is not an image')]), 'junk.jpg');
        fd.append('probe', JSON.stringify([embedding]));

        const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
        assert.equal(res.status, 400);
        assert.equal((await res.json()).error, 'unreadable_image');

        assert.deepEqual(uploadsSnapshot(), before,
            'multer temp file must be unlinked on the unreadable-image path');
    });
});

test('no target field present is rejected with 400 unreadable_image', async () => {
    await withServer(async (base) => {
        const fd = new FormData();
        fd.append('probe', JSON.stringify([Array.from({ length: 512 }, () => 0)]));
        const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
        assert.equal(res.status, 400);
        assert.equal((await res.json()).error, 'unreadable_image');
    });
});

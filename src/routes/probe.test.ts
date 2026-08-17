import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import path from 'path';
import { createMatcher, type Matcher } from '../face/matcher.js';
import { createApp } from '../app.js';

const IMAGES = path.join(process.cwd(), 'spike', 'images');

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
        server.close();
    }
}

function form(field: string, file: string) {
    const fd = new FormData();
    fd.append(field, new Blob([fs.readFileSync(path.join(IMAGES, file))]), file);
    return fd;
}

test('POST /probe returns a 512-d embedding and a relative box', async () => {
    await withServer(async (base) => {
        const res = await fetch(`${base}/probe`, {
            method: 'POST', body: form('source', 'obama_portrait.jpg'),
        });
        assert.equal(res.status, 200);
        const body = await res.json();
        assert.equal(body.embedding.length, 512);
        assert.ok(body.face.score > 0.5);
        assert.ok(body.face.box.width > 0 && body.face.box.width <= 1);
    });
});

test('POST /probe returns 422 when there is no face', async () => {
    await withServer(async (base) => {
        const blank = await (await import('sharp')).default({
            create: { width: 200, height: 200, channels: 3, background: '#888' },
        }).jpeg().toBuffer();
        const fd = new FormData();
        fd.append('source', new Blob([blank]), 'blank.jpg');

        const res = await fetch(`${base}/probe`, { method: 'POST', body: fd });
        assert.equal(res.status, 422);
        assert.equal((await res.json()).error, 'no_face_detected');
    });
});

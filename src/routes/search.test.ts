import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import fsp from 'fs/promises';
import os from 'os';
import path from 'path';
import { createMatcher, type Matcher } from '../face/matcher.js';
import { createStore } from '../people/store.js';
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

async function withServer(fn: (base: string) => Promise<void>, seed?: (dir: string) => Promise<void>) {
    // Most of these tests don't exercise the people store; a fresh temp dir
    // per call is only here because createApp requires one. `seed`, when
    // given, runs after the temp dir exists but before createStore reads it
    // -- letting a test plant a people.json the store's own API cannot
    // produce (e.g. a person with zero references) to prove /search doesn't
    // depend on that invariant holding elsewhere.
    const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'search-test-'));
    if (seed) await seed(dir);
    const app = createApp(await getMatcher(), await createStore(dir));
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

test('searches using a saved person id', async () => {
    await withServer(async (base) => {
        const fd = new FormData();
        fd.append('photo', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_portrait.jpg'))]), 'p.jpg');
        fd.append('name', 'Barack');
        const person = await (await fetch(`${base}/people`, { method: 'POST', body: fd })).json();

        const search = new FormData();
        search.append('target', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_cabinet.jpg'))]), 't.jpg');
        search.append('personId', person.id);

        const res = await fetch(`${base}/search`, { method: 'POST', body: search });
        assert.equal(res.status, 200);
        const body = await res.json();
        assert.equal(body.matched, true);
        assert.equal(body.faces.filter((f: any) => f.matched).length, 1);
    });
});

test('an unknown personId is rejected', async () => {
    await withServer(async (base) => {
        const fd = new FormData();
        fd.append('target', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_alt.jpg'))]), 't.jpg');
        fd.append('personId', 'p_nope');
        const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
        assert.equal(res.status, 400);
        assert.equal((await res.json()).error, 'bad_probe');
    });
});

// Closes a gap Task 5's review found: every route test up to this point
// sends a single-element `probe` array, so an implementation that silently
// truncated to refs[0] -- dropping every reference past the first -- would
// pass the entire route suite. This test discriminates on THREE
// independent axes, any one of which a refs[0]-only implementation fails:
//
//  1. `matched === true` at all: reference 0 is deliberately a photo of a
//     DIFFERENT person (Biden). Measured offline, cosine(obama_alt vs
//     biden_portrait) ~= -0.04 -- far below the 0.4 threshold -- while
//     cosine(obama_alt vs obama_portrait) ~= 0.76. A refs[0]-only
//     implementation would score every face against Biden alone and report
//     no match whatsoever.
//  2. `bestReference === 1` on the matched face: scoreFaces given only one
//     reference can never return an index other than 0, regardless of which
//     photo that reference is.
//  3. `cosine` on the matched face is well above the Biden-only ceiling
//     (~0), proving the reported score came from comparing against
//     reference 1, not reference 0.
//
// (No pair of *same-subject* Obama fixtures in spike/images/ works here --
// checked offline, every Obama-vs-Obama pair in this set scores 0.63-0.99,
// so none straddles the 0.4 threshold. A cross-identity reference 0 is what
// makes the discrimination airtight rather than a matter of degree.)
test('matches via the second reference only, proving max-over-references (not refs[0])', async () => {
    await withServer(async (base) => {
        const create = new FormData();
        create.append('photo', new Blob([fs.readFileSync(path.join(IMAGES, 'biden_portrait.jpg'))]), 'ref0.jpg');
        create.append('name', 'Cross-identity test person');
        const person = await (await fetch(`${base}/people`, { method: 'POST', body: create })).json();

        const addRef = new FormData();
        addRef.append('photo', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_portrait.jpg'))]), 'ref1.jpg');
        const updated = await (await fetch(`${base}/people/${person.id}/references`,
            { method: 'POST', body: addRef })).json();
        assert.equal(updated.references.length, 2, 'setup: person must have two references');

        const search = new FormData();
        search.append('target', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_alt.jpg'))]), 't.jpg');
        search.append('personId', person.id);
        const res = await fetch(`${base}/search`, { method: 'POST', body: search });

        assert.equal(res.status, 200);
        const body = await res.json();
        assert.equal(body.matched, true,
            'target only resembles reference 1 (Obama); a refs[0]-only impl would report no match');
        const matchedFaces = body.faces.filter((f: any) => f.matched);
        assert.equal(matchedFaces.length, 1);
        assert.equal(matchedFaces[0].bestReference, 1,
            'a refs[0]-only impl can only ever report bestReference 0');
        assert.ok(matchedFaces[0].cosine > 0.6,
            `expected a high cosine against the Obama reference, got ${matchedFaces[0].cosine}`);
    });
});

test('a saved person with zero references is rejected as a bad probe, not scored with empty refs', async () => {
    await withServer(async (base) => {
        const fd = new FormData();
        fd.append('target', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_alt.jpg'))]), 't.jpg');
        fd.append('personId', 'p_empty');
        const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
        assert.equal(res.status, 400);
        assert.equal((await res.json()).error, 'bad_probe');
    }, async (dir) => {
        // Plants a state the store's own API can never produce (it refuses
        // to delete a person's last reference) -- proving /search doesn't
        // depend on that invariant holding elsewhere. Without the guard,
        // scoreFaces([], ...) would return bestReference: -1, cosine:
        // -Infinity, and JSON.stringify(-Infinity) serializes to `null`,
        // silently shipping cosine: null on every face in the response.
        await fsp.writeFile(path.join(dir, 'people.json'), JSON.stringify({
            pipelineVersion: 1,
            people: [{ id: 'p_empty', name: 'Empty', createdAt: new Date().toISOString(), references: [] }],
        }));
    });
});

test('personId takes precedence over probe when both are sent', async () => {
    await withServer(async (base) => {
        // The saved person is Obama; the probe field carries an all-zero
        // (garbage) embedding that would never match anything. personId
        // winning proves it, not a fallback to probe.
        const create = new FormData();
        create.append('photo', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_portrait.jpg'))]), 'p.jpg');
        create.append('name', 'Precedence test person');
        const person = await (await fetch(`${base}/people`, { method: 'POST', body: create })).json();

        const fd = new FormData();
        fd.append('target', new Blob([fs.readFileSync(path.join(IMAGES, 'obama_alt.jpg'))]), 't.jpg');
        fd.append('personId', person.id);
        fd.append('probe', JSON.stringify([Array.from({ length: 512 }, () => 0)]));

        const res = await fetch(`${base}/search`, { method: 'POST', body: fd });
        assert.equal(res.status, 200);
        const body = await res.json();
        assert.equal(body.matched, true, 'personId must win over the garbage probe payload');
    });
});

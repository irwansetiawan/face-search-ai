import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import fsp from 'fs/promises';
import os from 'os';
import path from 'path';
import { createMatcher, type Matcher } from '../face/matcher.js';
import { createStore, type Store } from '../people/store.js';
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

async function tmpStore(): Promise<Store> {
    const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'people-routes-'));
    return createStore(dir);
}

async function withServer(fn: (base: string, store: Store) => Promise<void>) {
    const store = await tmpStore();
    const app = createApp(await getMatcher(), store);
    const server = app.listen(0);
    await new Promise(r => server.once('listening', r));
    const { port } = server.address() as { port: number };
    try {
        await fn(`http://127.0.0.1:${port}`, store);
    } finally {
        await new Promise<void>(resolve => server.close(() => resolve()));
    }
}

function photoForm(file: string, fields: Record<string, string> = {}) {
    const fd = new FormData();
    fd.append('photo', new Blob([fs.readFileSync(path.join(IMAGES, file))]), file);
    for (const [k, v] of Object.entries(fields)) fd.append(k, v);
    return fd;
}

async function createPerson(base: string, file: string, name: string) {
    const res = await fetch(`${base}/people`, { method: 'POST', body: photoForm(file, { name }) });
    return { status: res.status, body: await res.json() };
}

test('creates a person, lists them, and adds a second reference', async () => {
    await withServer(async (base) => {
        const created = await fetch(`${base}/people`, {
            method: 'POST', body: photoForm('obama_portrait.jpg', { name: 'Barack' }),
        });
        assert.equal(created.status, 201);
        const person = await created.json();
        assert.equal(person.name, 'Barack');
        assert.equal(person.references.length, 1);
        assert.ok(person.references[0].thumb, 'expected an avatar path');
        assert.equal(person.references[0].embedding, undefined,
            'the 512-float embedding must never be shipped to the browser');

        const listed = await (await fetch(`${base}/people`)).json();
        assert.equal(listed.length, 1);
        assert.ok(listed[0].references[0].thumb, 'expected an avatar path');
        assert.equal(listed[0].references[0].embedding, undefined);

        const added = await fetch(`${base}/people/${person.id}/references`, {
            method: 'POST', body: photoForm('obama_alt.jpg'),
        });
        assert.equal(added.status, 200);
        const updated = await added.json();
        assert.equal(updated.references.length, 2);
    });
});

test('POST /:id/references: a server-side write fault is also a 500, not 400 unreadable_image', async () => {
    await withServer(async (base, store) => {
        const { body: person } = await createPerson(base, 'obama_portrait.jpg', 'Barack');
        const personDir = path.join(store.dataDir, 'people', person.id);
        await fsp.chmod(personDir, 0o500); // read+execute only, no write
        try {
            const res = await fetch(`${base}/people/${person.id}/references`, {
                method: 'POST', body: photoForm('obama_alt.jpg'),
            });
            assert.equal(res.status, 500,
                'a disk/permission fault writing artefacts must not be blamed on the client');
            assert.notEqual((await res.json()).error, 'unreadable_image');
        } finally {
            await fsp.chmod(personDir, 0o700);
        }
    });
});

test('rejects a photo with no face', async () => {
    await withServer(async (base) => {
        const blank = await (await import('sharp')).default({
            create: { width: 200, height: 200, channels: 3, background: '#888' },
        }).jpeg().toBuffer();
        const fd = new FormData();
        fd.append('photo', new Blob([blank]), 'blank.jpg');
        fd.append('name', 'Nobody');

        const res = await fetch(`${base}/people`, { method: 'POST', body: fd });
        assert.equal(res.status, 422);
        assert.equal((await res.json()).error, 'no_face_detected');

        // Nothing half-registered: no person should have been created.
        assert.equal((await (await fetch(`${base}/people`)).json()).length, 0);
    });
});

test('deletes a person', async () => {
    await withServer(async (base) => {
        const person = (await createPerson(base, 'obama_portrait.jpg', 'Barack')).body;

        const del = await fetch(`${base}/people/${person.id}`, { method: 'DELETE' });
        assert.equal(del.status, 204);
        assert.equal((await (await fetch(`${base}/people`)).json()).length, 0);
    });
});

test('POST /people requires both a name and a photo, and cleans up the temp upload either way', async () => {
    await withServer(async (base) => {
        const noName = await fetch(`${base}/people`, {
            method: 'POST', body: photoForm('obama_portrait.jpg'),
        });
        assert.equal(noName.status, 400);
        assert.equal((await noName.json()).error, 'name_and_photo_required');

        const fd = new FormData();
        fd.append('name', 'No Photo');
        const noPhoto = await fetch(`${base}/people`, { method: 'POST', body: fd });
        assert.equal(noPhoto.status, 400);
        assert.equal((await noPhoto.json()).error, 'name_and_photo_required');

        assert.equal((await (await fetch(`${base}/people`)).json()).length, 0);
    });
});

test('unreadable bytes are rejected with 400 unreadable_image, not registered as a person', async () => {
    await withServer(async (base) => {
        const fd = new FormData();
        fd.append('photo', new Blob([Buffer.from('this is not an image')]), 'junk.jpg');
        fd.append('name', 'Junk');

        const res = await fetch(`${base}/people`, { method: 'POST', body: fd });
        assert.equal(res.status, 400);
        assert.equal((await res.json()).error, 'unreadable_image');
        assert.equal((await (await fetch(`${base}/people`)).json()).length, 0);
    });
});

test('a server-side write fault (disk/permissions) is a 500, not 400 unreadable_image', async () => {
    await withServer(async (base, store) => {
        // The uploaded photo is perfectly decodable; what fails is writing
        // the stored artefacts (mkdir inside writeArtifacts), which is a
        // server-side fault, not the client's. This is the regression
        // finding 1 covers: the try/catch that maps a throw to 400
        // unreadable_image must be scoped to the decode step only, not to
        // everything that happens afterward.
        const peopleDir = path.join(store.dataDir, 'people');
        await fsp.chmod(peopleDir, 0o500); // read+execute only, no write
        try {
            const res = await fetch(`${base}/people`, {
                method: 'POST', body: photoForm('obama_portrait.jpg', { name: 'Barack' }),
            });
            assert.equal(res.status, 500,
                'a disk/permission fault writing artefacts must not be blamed on the client');
            assert.notEqual((await res.json()).error, 'unreadable_image');
        } finally {
            await fsp.chmod(peopleDir, 0o700);
        }
    });
});

test('a store flush failure during person creation is a 500 and leaves no orphaned pending directory', async () => {
    await withServer(async (base, store) => {
        // Unlike the write-fault test above (which fails inside
        // writeArtifacts, before any directory is ever registered with the
        // store), this fails one step later: writeArtifacts succeeds (both
        // artefacts land under people/pending_<hex>/), but
        // store.addPerson's flush() -- which writes people.json directly
        // under dataDir, not under dataDir/people -- hits EACCES. `people/`
        // itself keeps its normal mode, so mkdir and both artefact writes
        // still succeed.
        await fsp.chmod(store.dataDir, 0o500); // read+execute only, no write
        try {
            const res = await fetch(`${base}/people`, {
                method: 'POST', body: photoForm('obama_portrait.jpg', { name: 'Barack' }),
            });
            assert.equal(res.status, 500,
                'a disk/permission fault persisting the new person must not be blamed on the client');

            const entries = await fsp.readdir(path.join(store.dataDir, 'people'));
            assert.deepEqual(entries, [],
                'writeArtifacts succeeded but addPerson never registered it -- ' +
                'the pending directory must be cleaned up, not left orphaned');
        } finally {
            await fsp.chmod(store.dataDir, 0o700);
        }
    });
});

test('POST /people leaves no orphaned pending directory behind on success', async () => {
    await withServer(async (base, store) => {
        const { body: person } = await createPerson(base, 'obama_portrait.jpg', 'Barack');
        const entries = await fsp.readdir(path.join(store.dataDir, 'people'));
        assert.deepEqual(entries, [person.id],
            'the files must live under people/<personId>/, not a leftover pending_ directory');
    });
});

test('people.json (which carries every 512-float embedding) is not reachable through /people-files', async () => {
    await withServer(async (base) => {
        await createPerson(base, 'obama_portrait.jpg', 'Barack');
        const res = await fetch(`${base}/people-files/people.json`);
        assert.equal(res.status, 404,
            'the static mount must not expose the whole data directory, only people/<id>/ artefacts');
    });
});

test('adding a reference to an unknown person is 404, not a silent create', async () => {
    await withServer(async (base) => {
        const res = await fetch(`${base}/people/p_doesnotexist/references`, {
            method: 'POST', body: photoForm('obama_portrait.jpg'),
        });
        assert.equal(res.status, 404);
        assert.equal((await res.json()).error, 'no_such_person');
    });
});

test('deleting an unknown person is 404, not the store\'s silent 204 no-op', async () => {
    await withServer(async (base) => {
        const res = await fetch(`${base}/people/p_doesnotexist`, { method: 'DELETE' });
        assert.equal(res.status, 404);
        assert.equal((await res.json()).error, 'no_such_person');
    });
});

test('deleting a reference from an unknown person is 404', async () => {
    await withServer(async (base) => {
        const res = await fetch(`${base}/people/p_doesnotexist/references/ref_x`, { method: 'DELETE' });
        assert.equal(res.status, 404);
        assert.equal((await res.json()).error, 'no_such_person');
    });
});

test('deleting an unknown reference id on a real person is 404', async () => {
    await withServer(async (base) => {
        const { body: person } = await createPerson(base, 'obama_portrait.jpg', 'Barack');
        const res = await fetch(`${base}/people/${person.id}/references/ref_doesnotexist`, { method: 'DELETE' });
        assert.equal(res.status, 404);
        assert.equal((await res.json()).error, 'no_such_reference');
    });
});

test('deleting the last reference of a person is refused, not silently emptying it', async () => {
    await withServer(async (base) => {
        const { body: person } = await createPerson(base, 'obama_portrait.jpg', 'Barack');
        const refId = person.references[0].id;

        const res = await fetch(`${base}/people/${person.id}/references/${refId}`, { method: 'DELETE' });
        assert.ok(res.status >= 400 && res.status < 500,
            `expected a client-error status refusing the deletion, got ${res.status}`);

        const stillThere = await (await fetch(`${base}/people`)).json();
        assert.equal(stillThere.length, 1);
        assert.equal(stillThere[0].references.length, 1);
    });
});

test('deleting a non-last reference succeeds and leaves the other one intact', async () => {
    await withServer(async (base) => {
        const { body: person } = await createPerson(base, 'obama_portrait.jpg', 'Barack');
        const added = await (await fetch(`${base}/people/${person.id}/references`, {
            method: 'POST', body: photoForm('obama_alt.jpg'),
        })).json();
        assert.equal(added.references.length, 2);
        const [first, second] = added.references;

        const res = await fetch(`${base}/people/${person.id}/references/${first.id}`, { method: 'DELETE' });
        assert.equal(res.status, 200);
        const body = await res.json();
        assert.equal(body.references.length, 1);
        assert.equal(body.references[0].id, second.id);
    });
});

test('the saved avatar is cropped from the same face that was embedded, on a multi-face photo where the largest face is not the first detected', async () => {
    await withServer(async (base) => {
        const m = await getMatcher();
        const buffer = fs.readFileSync(path.join(IMAGES, 'g8_group.jpg'));
        const { faces } = await m.embedAll(buffer);
        const largest = faces.reduce((a, b) =>
            a.box.width * a.box.height >= b.box.width * b.box.height ? a : b);

        // Sanity check on the fixture itself: this test only proves anything
        // if the largest face is NOT the first one detected. If this fails,
        // the fixture stopped exercising the bug this test exists to catch.
        assert.notEqual(largest.faceIndex, 0,
            'fixture must have its largest face at a non-zero detection index');

        const { status, body: person } = await createPerson(base, 'g8_group.jpg', 'G8');
        assert.equal(status, 201);

        const expectedThumb = await m.alignedCropPng(buffer, largest.faceIndex);
        const wrongThumb = await m.alignedCropPng(buffer, 0);

        const thumbRes = await fetch(`${base}/people-files/${person.references[0].thumb}`);
        assert.equal(thumbRes.status, 200, 'the stored avatar must be servable via /people-files');
        const actualThumb = Buffer.from(await thumbRes.arrayBuffer());

        assert.deepEqual(actualThumb, expectedThumb,
            'avatar must be cropped from the same face index the embedding came from (face.faceIndex)');
        assert.notDeepEqual(actualThumb, wrongThumb,
            'a hardcoded index 0 would produce a different crop than the one actually embedded');
    });
});

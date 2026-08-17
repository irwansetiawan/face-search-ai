import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { createStore } from './store.js';

async function tmpDir(): Promise<string> {
    return fs.mkdtemp(path.join(os.tmpdir(), 'people-'));
}

const ref = (fill: number) => ({
    image: 'people/x/ref-1.jpg',
    thumb: 'people/x/ref-1.thumb.png',
    embedding: new Array(512).fill(fill),
    detScore: 0.9,
});

test('adds a person and persists across reloads', async () => {
    const dir = await tmpDir();
    const store = await createStore(dir);
    const person = await store.addPerson('Sarah', ref(0.1));

    assert.equal(person.name, 'Sarah');
    assert.equal(person.references.length, 1);

    const reloaded = await createStore(dir);
    assert.equal(reloaded.list().length, 1);
    assert.equal(reloaded.get(person.id)!.name, 'Sarah');
});

test('supports multiple references per person', async () => {
    const store = await createStore(await tmpDir());
    const person = await store.addPerson('Sarah', ref(0.1));
    const updated = await store.addReference(person.id, ref(0.2));
    assert.equal(updated.references.length, 2);
    assert.notEqual(updated.references[0].id, updated.references[1].id);
});

test('refuses to delete the last reference', async () => {
    const store = await createStore(await tmpDir());
    const person = await store.addPerson('Sarah', ref(0.1));
    await assert.rejects(
        () => store.deleteReference(person.id, person.references[0].id),
        /last reference/
    );
});

test('deletes a non-last reference, and it stays gone on reload', async () => {
    const dir = await tmpDir();
    const store = await createStore(dir);
    const person = await store.addPerson('Sarah', ref(0.1));
    const second = await store.addReference(person.id, ref(0.2));
    const keptId = second.references[0].id;
    const removedId = second.references[1].id;

    const updated = await store.deleteReference(person.id, removedId);
    assert.equal(updated.references.length, 1);
    assert.equal(updated.references[0].id, keptId);

    const reloaded = await createStore(dir);
    assert.equal(reloaded.get(person.id)!.references.length, 1);
    assert.equal(reloaded.get(person.id)!.references[0].id, keptId);
});

test('a rejected mutation does not poison the queue for later calls', async () => {
    const store = await createStore(await tmpDir());
    const person = await store.addPerson('Sarah', ref(0.1));

    await assert.rejects(
        () => store.deleteReference(person.id, person.references[0].id),
        /last reference/
    );

    // If the earlier rejection had poisoned the internal serialization
    // queue, this call would hang or reject too.
    const updated = await store.addReference(person.id, ref(0.2));
    assert.equal(updated.references.length, 2);
});

test('replaceAll overwrites the store and persists across reload', async () => {
    const dir = await tmpDir();
    const store = await createStore(dir);
    await store.addPerson('Sarah', ref(0.1));

    const replacement = [
        { id: 'p_zzz', name: 'Zoe', createdAt: new Date().toISOString(), references: [{ ...ref(0.4), id: 'ref_zzz', addedAt: new Date().toISOString() }] },
    ];
    await store.replaceAll(replacement);

    assert.equal(store.list().length, 1);
    assert.equal(store.list()[0].name, 'Zoe');

    const reloaded = await createStore(dir);
    assert.equal(reloaded.list().length, 1);
    assert.equal(reloaded.list()[0].name, 'Zoe');
});

test('deletes a person', async () => {
    const dir = await tmpDir();
    const store = await createStore(dir);
    const person = await store.addPerson('Sarah', ref(0.1));
    await store.deletePerson(person.id);
    assert.equal(store.list().length, 0);
    assert.equal((await createStore(dir)).list().length, 0);
});

test('deleting a person also removes its image directory', async () => {
    const dir = await tmpDir();
    const store = await createStore(dir);
    const person = await store.addPerson('Sarah', ref(0.1));

    const personDir = path.join(dir, 'people', person.id);
    await fs.mkdir(personDir, { recursive: true });
    await fs.writeFile(path.join(personDir, 'ref-1.jpg'), 'fake image bytes');

    await store.deletePerson(person.id);

    await assert.rejects(() => fs.access(personDir));
});

test('a corrupt people.json causes createStore to reject rather than starting empty', async () => {
    const dir = await tmpDir();
    await fs.mkdir(path.join(dir, 'people'), { recursive: true });
    await fs.writeFile(path.join(dir, 'people.json'), '{not valid json');

    await assert.rejects(() => createStore(dir));
});

test('an existing valid store is still intact on disk after a failed createStore', async () => {
    const dir = await tmpDir();
    const store = await createStore(dir);
    await store.addPerson('Sarah', ref(0.1));

    const goodContents = await fs.readFile(path.join(dir, 'people.json'), 'utf8');
    assert.match(goodContents, /Sarah/);

    // Corrupt the file out from under the store, as if it had been
    // truncated by an earlier crash or edited externally.
    await fs.writeFile(path.join(dir, 'people.json'), '{"people": [ this is not json');

    await assert.rejects(() => createStore(dir));

    // The critical assertion: createStore must not have overwritten the
    // file with an empty store on its way to rejecting. Whatever is on
    // disk must still be the corrupt content we wrote, untouched.
    const afterFailedLoad = await fs.readFile(path.join(dir, 'people.json'), 'utf8');
    assert.equal(afterFailedLoad, '{"people": [ this is not json');
});

test('updateReferencePaths corrects one reference in place without stamping pipelineVersion', async () => {
    const dir = await tmpDir();
    const store = await createStore(dir);
    const person = await store.addPerson('Sarah', ref(0.1));
    const refId = person.references[0].id;

    // Seed a stale on-disk version directly, bypassing the store, so we can
    // tell whether this method touches it.
    const before = JSON.parse(await fs.readFile(path.join(dir, 'people.json'), 'utf8'));
    before.pipelineVersion = before.pipelineVersion - 1;
    await fs.writeFile(path.join(dir, 'people.json'), JSON.stringify(before, null, 2));
    const reloaded = await createStore(dir);

    const updated = await reloaded.updateReferencePaths(person.id, refId, 'people/x/new.jpg', 'people/x/new.thumb.png');
    assert.equal(updated.references[0].image, 'people/x/new.jpg');
    assert.equal(updated.references[0].thumb, 'people/x/new.thumb.png');

    const onDisk = JSON.parse(await fs.readFile(path.join(dir, 'people.json'), 'utf8'));
    assert.equal(onDisk.people[0].references[0].image, 'people/x/new.jpg');
    assert.equal(onDisk.pipelineVersion, before.pipelineVersion,
        'updateReferencePaths must not stamp pipelineVersion current');
});

test('updateReferenceEmbedding corrects one reference in place without stamping pipelineVersion', async () => {
    const dir = await tmpDir();
    const store = await createStore(dir);
    const person = await store.addPerson('Sarah', ref(0.1));
    const refId = person.references[0].id;

    const before = JSON.parse(await fs.readFile(path.join(dir, 'people.json'), 'utf8'));
    before.pipelineVersion = before.pipelineVersion - 1;
    await fs.writeFile(path.join(dir, 'people.json'), JSON.stringify(before, null, 2));
    const reloaded = await createStore(dir);

    const newEmbedding = new Array(512).fill(0.9);
    const updated = await reloaded.updateReferenceEmbedding(person.id, refId, newEmbedding, 0.77);
    assert.deepEqual(updated.references[0].embedding, newEmbedding);
    assert.equal(updated.references[0].detScore, 0.77);

    const onDisk = JSON.parse(await fs.readFile(path.join(dir, 'people.json'), 'utf8'));
    assert.deepEqual(onDisk.people[0].references[0].embedding, newEmbedding);
    assert.equal(onDisk.pipelineVersion, before.pipelineVersion,
        'updateReferenceEmbedding must not stamp pipelineVersion current');
});

test('updateReferencePaths on an unknown reference id rejects, and on an unknown person rejects', async () => {
    const store = await createStore(await tmpDir());
    const person = await store.addPerson('Sarah', ref(0.1));

    await assert.rejects(
        () => store.updateReferencePaths(person.id, 'ref_nope', 'a', 'b'),
        /no such reference/
    );
    await assert.rejects(
        () => store.updateReferencePaths('p_nope', person.references[0].id, 'a', 'b'),
        /no such person/
    );
});

test('concurrent writes do not lose entries', async () => {
    const dir = await tmpDir();
    const store = await createStore(dir);
    await Promise.all([
        store.addPerson('A', ref(0.1)),
        store.addPerson('B', ref(0.2)),
        store.addPerson('C', ref(0.3)),
    ]);
    assert.equal((await createStore(dir)).list().length, 3);
});

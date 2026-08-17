import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { createStore, PIPELINE_VERSION } from './store.js';
import { reembedIfStale } from './reembed.js';
import { createMatcher, type Matcher } from '../face/matcher.js';

const SRC = path.join(process.cwd(), 'spike', 'images', 'obama_portrait.jpg');

// Ruling 14: share one matcher per test file. createMatcher() compiles the
// ONNX graphs (~5s), so creating it per-test would make the suite
// progressively slower across the whole plan. The matcher holds no
// per-request state beyond its serialization queue, so sharing it across
// tests in this file is safe.
let matcher: Matcher | undefined;
async function getMatcher(): Promise<Matcher> {
    return matcher ??= await createMatcher();
}

async function tmpDir(): Promise<string> {
    return fs.mkdtemp(path.join(os.tmpdir(), 'reembed-'));
}

/** Places a readable copy of the fixture photo under `dir` and returns its
 * store-relative path. */
async function plantImage(dir: string): Promise<string> {
    const rel = path.join('people', 'tmp', 'ref-1.jpg');
    await fs.mkdir(path.join(dir, 'people', 'tmp'), { recursive: true });
    await fs.copyFile(SRC, path.join(dir, rel));
    return rel;
}

test('stale stored embeddings are recomputed from the stored image and persisted', async () => {
    const m = await getMatcher();
    const dir = await tmpDir();
    const store = await createStore(dir);
    const rel = await plantImage(dir);

    // A deliberately wrong embedding and detScore, as a stale one would be.
    const person = await store.addPerson('Sarah', {
        image: rel, thumb: rel, embedding: new Array(512).fill(0), detScore: 0.111,
    });

    const count = await reembedIfStale(store, m, PIPELINE_VERSION - 1);
    assert.equal(count, 1);

    const fixedRef = store.get(person.id)!.references[0];
    const fixed = fixedRef.embedding;

    // Ground truth: embed the same source image directly and compare. This
    // is the assertion that would fail if the re-embed produced *some*
    // normalized-looking vector that wasn't actually derived correctly from
    // the stored image (e.g. wrong file, wrong crop, a bug in the
    // Float32Array -> number[] conversion) — a norm check alone would not
    // catch any of that.
    const groundTruth = await m.embedLargest(await fs.readFile(SRC));
    assert.ok(groundTruth, 'sanity: the fixture image must contain a detectable face');
    const dot = fixed.reduce((sum, v, i) => sum + v * groundTruth!.embedding[i], 0);
    assert.ok(dot > 0.999, `expected the re-embedded vector to match a direct embed of the same image, cosine similarity was ${dot}`);

    const norm = Math.sqrt(fixed.reduce((a, v) => a + v * v, 0));
    assert.ok(Math.abs(norm - 1) < 1e-3, `expected an L2-normalized vector, norm was ${norm}`);

    // detScore must be refreshed alongside the embedding, not left stale.
    assert.equal(fixedRef.detScore, groundTruth!.detScore);
    assert.notEqual(fixedRef.detScore, 0.111);

    // Persistence: reload from disk, not just the live in-memory Person, so
    // a bug that skipped replaceAll (or the number[] round-trip through
    // JSON) would be caught rather than passing on live state alone.
    const reloaded = await createStore(dir);
    const persisted = reloaded.get(person.id)!.references[0].embedding;
    const pdot = persisted.reduce((sum, v, i) => sum + v * groundTruth!.embedding[i], 0);
    assert.ok(pdot > 0.999, `persisted vector lost fidelity through JSON, cosine similarity was ${pdot}`);

    const onDisk = JSON.parse(await fs.readFile(path.join(dir, 'people.json'), 'utf8'));
    assert.equal(onDisk.pipelineVersion, PIPELINE_VERSION,
        'must stamp the current version, or every boot would re-embed again');
});

test('a reference whose image file is missing keeps its stale vector instead of being dropped, and the skip is logged', async () => {
    const m = await getMatcher();
    const dir = await tmpDir();
    const store = await createStore(dir);

    const staleEmbedding = new Array(512).fill(0);
    const relImage = path.join('people', 'nope', 'missing.jpg');
    const person = await store.addPerson('Ghost', {
        image: relImage, thumb: 'missing.jpg',
        embedding: staleEmbedding, detScore: 0.5,
    });

    const warnings: unknown[][] = [];
    const originalWarn = console.warn;
    console.warn = (...args: unknown[]) => { warnings.push(args); };
    let count: number;
    try {
        count = await reembedIfStale(store, m, PIPELINE_VERSION - 1);
    } finally {
        console.warn = originalWarn;
    }
    assert.equal(count, 0, 'a missing file should not count as re-embedded');

    const stillThere = store.get(person.id);
    assert.ok(stillThere, 'the person must not be dropped when its image is unreadable');
    assert.deepEqual(stillThere!.references[0].embedding, staleEmbedding);
    assert.equal(stillThere!.references[0].detScore, 0.5);

    // The skip must be visible in boot output: a per-reference warning
    // naming the person id, reference id, and the failing image path, plus
    // a summary line. Silence here is exactly the failure mode this
    // recovery mechanism must not have — a stale reference could otherwise
    // be skipped forever, on every future boot, with no signal anywhere.
    const refId = stillThere!.references[0].id;
    const perReference = warnings.find(args =>
        typeof args[0] === 'string' &&
        args[0].includes(person.id) &&
        args[0].includes(refId) &&
        args[0].includes(relImage)
    );
    assert.ok(perReference, `expected a warning naming person ${person.id}, reference ${refId}, and path ${relImage}; got: ${JSON.stringify(warnings)}`);

    const summary = warnings.find(args =>
        typeof args[0] === 'string' && /\b1\b.*skipped|skipped.*\b1\b/.test(args[0])
    );
    assert.ok(summary, `expected a summary line mentioning 1 skipped reference; got: ${JSON.stringify(warnings)}`);
});

test('a partial failure persists the successful re-embeds but leaves pipelineVersion stale, so the failed reference retries on the next boot', async () => {
    const m = await getMatcher();
    const dir = await tmpDir();
    const goodRel = await plantImage(dir);
    const missingRel = path.join('people', 'nope', 'missing.jpg');
    const personId = 'p_1';
    const goodRefId = 'ref_good';
    const missingRefId = 'ref_missing';

    // Seed the on-disk file directly, the same way the "nothing to update"
    // test below does. This matters here specifically: a freshly created
    // store's very first addPerson/addReference flush already writes
    // pipelineVersion: PIPELINE_VERSION (the in-memory default for a store
    // with no prior file), so building this fixture through store.addPerson
    // calls would make the file "current" from the moment of creation --
    // masking the exact distinction (stale vs. current on disk) this test
    // exists to prove.
    await fs.mkdir(path.join(dir, 'people'), { recursive: true });
    const seed = {
        pipelineVersion: PIPELINE_VERSION - 1,
        people: [{
            id: personId, name: 'Sarah', createdAt: new Date().toISOString(),
            references: [
                {
                    id: goodRefId, image: goodRel, thumb: goodRel,
                    embedding: new Array(512).fill(0), detScore: 0.111, addedAt: new Date().toISOString(),
                },
                {
                    id: missingRefId, image: missingRel, thumb: missingRel,
                    embedding: new Array(512).fill(0), detScore: 0.222, addedAt: new Date().toISOString(),
                },
            ],
        }],
    };
    await fs.writeFile(path.join(dir, 'people.json'), JSON.stringify(seed, null, 2));
    const store = await createStore(dir);

    const groundTruth = await m.embedLargest(await fs.readFile(SRC));
    assert.ok(groundTruth, 'sanity: the fixture image must contain a detectable face');

    // First call ("first boot"): one reference re-embeds cleanly, one is
    // missing and skipped. This is a PARTIAL run.
    const firstCount = await reembedIfStale(store, m, PIPELINE_VERSION - 1);
    assert.equal(firstCount, 1, 'exactly the valid reference should have re-embedded');

    let onDisk = JSON.parse(await fs.readFile(path.join(dir, 'people.json'), 'utf8'));
    const persistedGood = onDisk.people
        .find((p: { id: string }) => p.id === personId).references
        .find((r: { id: string }) => r.id === goodRefId);
    const dot = persistedGood.embedding.reduce(
        (sum: number, v: number, i: number) => sum + v * groundTruth!.embedding[i], 0);
    assert.ok(dot > 0.999,
        'the successfully re-embedded reference must be persisted to disk even though the run as a whole was partial');

    // The assertion this test exists for: a partial run must NOT stamp
    // pipelineVersion current. Doing so (the pre-fix behavior, inherited
    // from replaceAll(people) firing whenever updated > 0) would make the
    // still-stale, still-unfixed reference permanently un-retryable -- the
    // store would look "current" forever despite one reference silently
    // keeping a stale embedding.
    assert.equal(onDisk.pipelineVersion, PIPELINE_VERSION - 1,
        'a partial run must leave pipelineVersion stale, or the skipped reference would never retry again');

    // Second call ("next boot"): storedVersion is read from disk again --
    // still stale, because of the assertion just above -- so the whole
    // store is retried, including the reference that failed last time. This
    // is the assertion that actually proves the retry guarantee holds, not
    // just that the version stamp looks right in isolation.
    const warnings: unknown[][] = [];
    const originalWarn = console.warn;
    console.warn = (...args: unknown[]) => { warnings.push(args); };
    let secondCount: number;
    try {
        secondCount = await reembedIfStale(store, m, onDisk.pipelineVersion);
    } finally {
        console.warn = originalWarn;
    }
    assert.equal(secondCount, 1, 'the valid reference re-embeds again on retry (redundant, but harmless)');

    const retriedMissingOne = warnings.find(args =>
        typeof args[0] === 'string' &&
        args[0].includes(personId) &&
        args[0].includes(missingRefId));
    assert.ok(retriedMissingOne,
        `expected the second call to retry the previously-failed reference ${missingRefId}; got: ${JSON.stringify(warnings)}`);

    onDisk = JSON.parse(await fs.readFile(path.join(dir, 'people.json'), 'utf8'));
    assert.equal(onDisk.pipelineVersion, PIPELINE_VERSION - 1,
        'still stale after the second call too, since the missing reference still fails every time');
});

test('a store with nothing to update does not write, and its stale version stamp survives', async () => {
    const m = await getMatcher();
    const dir = await tmpDir();

    // `flush()` deliberately preserves whatever pipelineVersion is already
    // on disk (see store.ts), so the only way to stamp PIPELINE_VERSION
    // forward is replaceAll. Seed a file that is stale but whose only
    // reference has a missing image, so re-embedding is attempted, fails,
    // and `updated` stays 0 — this exercises the `updated > 0` write gate
    // for real, rather than tautologically (a `replaceAll` with unchanged
    // people would otherwise write byte-identical JSON and this test
    // wouldn't be able to tell the difference).
    const file = path.join(dir, 'people.json');
    await fs.mkdir(path.join(dir, 'people'), { recursive: true });
    const seed = {
        pipelineVersion: PIPELINE_VERSION - 1,
        people: [{
            id: 'p_1', name: 'Ghost', createdAt: new Date().toISOString(),
            references: [{
                id: 'ref_1', image: path.join('people', 'nope', 'missing.jpg'), thumb: 'x.png',
                embedding: new Array(512).fill(0), detScore: 0.5, addedAt: new Date().toISOString(),
            }],
        }],
    };
    await fs.writeFile(file, JSON.stringify(seed, null, 2));

    const store = await createStore(dir);
    const count = await reembedIfStale(store, m, PIPELINE_VERSION - 1);
    assert.equal(count, 0);

    const onDisk = JSON.parse(await fs.readFile(file, 'utf8'));
    assert.equal(onDisk.pipelineVersion, PIPELINE_VERSION - 1,
        'nothing changed, so replaceAll must not have run, and the stale stamp must survive');
});

test('current-version stores are left completely untouched, even with a re-embeddable image', async () => {
    const m = await getMatcher();
    const dir = await tmpDir();
    const store = await createStore(dir);
    const rel = await plantImage(dir);

    // Deliberately use a *readable* image here, not a bad path: if the
    // version guard were ever removed or weakened, this reference would
    // actually get re-embedded and the file would change. A bad path would
    // let a missing guard hide behind "the file didn't exist anyway."
    const sentinel = new Array(512).fill(0);
    const person = await store.addPerson('Sarah', {
        image: rel, thumb: rel, embedding: sentinel, detScore: 0.9,
    });
    const before = await fs.readFile(path.join(dir, 'people.json'), 'utf8');

    const count = await reembedIfStale(store, m, PIPELINE_VERSION);
    assert.equal(count, 0, 'must not touch a current store, even with a re-embeddable image present');

    const stillSentinel = store.get(person.id)!.references[0];
    assert.deepEqual(stillSentinel.embedding, sentinel);
    assert.equal(stillSentinel.detScore, 0.9);

    const after = await fs.readFile(path.join(dir, 'people.json'), 'utf8');
    assert.equal(after, before, 'a current-version store must never be written to');
});

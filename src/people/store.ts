/**
 * Persists the saved-people library: names and 512-float embeddings, plus a
 * path to a downscaled copy of each reference photo. The photo itself is not
 * expensive to keep around, but re-deriving an embedding from scratch is —
 * so `pipelineVersion` and each reference's `image` path exist specifically
 * so a future pipeline fix can re-embed automatically (see Task 9's
 * `reembedIfStale`) rather than asking the user to re-upload every photo.
 *
 * This module owns storage only: no routes, no image writes, no re-embedding.
 */
import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';

export const PIPELINE_VERSION = 1;

export type Reference = {
    id: string;
    image: string;
    thumb: string;
    embedding: number[];
    detScore: number;
    addedAt: string;
};

export type Person = {
    id: string;
    name: string;
    createdAt: string;
    references: Reference[];
};

export type NewReference = Omit<Reference, 'id' | 'addedAt'>;

type FileShape = { pipelineVersion: number; people: Person[] };

export type Store = {
    list(): Person[];
    get(id: string): Person | undefined;
    addPerson(name: string, ref: NewReference): Promise<Person>;
    addReference(personId: string, ref: NewReference): Promise<Person>;
    deletePerson(id: string): Promise<void>;
    deleteReference(personId: string, refId: string): Promise<Person>;
    /** Corrects one reference's stored file paths in place. Does NOT stamp
     * pipelineVersion -- see the comment at its implementation. */
    updateReferencePaths(personId: string, refId: string, image: string, thumb: string): Promise<Person>;
    /** Corrects one reference's embedding/detScore in place. Does NOT stamp
     * pipelineVersion -- see the comment at its implementation. */
    updateReferenceEmbedding(personId: string, refId: string, embedding: number[], detScore: number): Promise<Person>;
    replaceAll(people: Person[]): Promise<void>;
    dataDir: string;
};

const shortId = (prefix: string) => `${prefix}_${crypto.randomBytes(3).toString('hex')}`;

export async function createStore(dataDir: string): Promise<Store> {
    const file = path.join(dataDir, 'people.json');
    await fs.mkdir(path.join(dataDir, 'people'), { recursive: true });

    // `state.pipelineVersion` deliberately comes from the file, not from the
    // PIPELINE_VERSION constant, except on first run. Task 9's staleness
    // check depends on this file-vs-code comparison: `flush()` must write
    // back whatever version is already in `state`, unchanged, so that a
    // stale file stays observably stale until `replaceAll` (the only place
    // that stamps PIPELINE_VERSION) re-embeds it. Do not "simplify" flush()
    // to always stamp PIPELINE_VERSION — that would silently defeat
    // staleness detection the moment any addPerson/addReference call fires.
    let state: FileShape = { pipelineVersion: PIPELINE_VERSION, people: [] };
    try {
        state = JSON.parse(await fs.readFile(file, 'utf8'));
        // A file missing the key entirely (an older format) must read as
        // stale (0), not as "current" -- `?? PIPELINE_VERSION` would
        // resolve a MISSING key to current, making an unstamped store
        // permanently invisible to Task 9's migration (and sticky, since
        // JSON.stringify drops an undefined key on every subsequent flush).
        // A non-number value (hand-edited file, future format change) must
        // not read as current either. This only normalizes what came off
        // disk -- the in-memory first-run default above is left untouched,
        // so a genuine first run still starts at PIPELINE_VERSION and never
        // looks stale.
        state.pipelineVersion = typeof state.pipelineVersion === 'number' ? state.pipelineVersion : 0;
    } catch (err) {
        // ENOENT is the only failure that means "first run, no file yet" —
        // start from the empty default above. Everything else (permissions,
        // EISDIR, a truncated/corrupt people.json, a JSON.parse SyntaxError
        // on malformed content) must NOT be treated as an empty store: the
        // very next mutation would flush() and atomically rename over the
        // still-present real file, permanently destroying every saved
        // person with nothing surfaced anywhere. Fail loudly instead.
        if ((err as NodeJS.ErrnoException).code !== 'ENOENT') {
            const reason = err instanceof Error ? err.message : String(err);
            throw new Error(
                `people store at ${file} could not be read (${reason}). ` +
                'Refusing to start with an empty store, which would overwrite ' +
                'existing data on the next write. Repair or move the file and retry.'
            );
        }
        // genuine first run: no file yet.
    }

    // Serialize mutations so a concurrent read-modify-write (including the
    // final write-to-disk) cannot interleave and lose an entry. `run` is
    // what callers await (and can reject on); `queue` swallows the
    // rejection so a failed call doesn't poison the chain for whoever
    // queues up next.
    let queue: Promise<unknown> = Promise.resolve();
    function serialize<T>(fn: () => Promise<T>): Promise<T> {
        const run = queue.then(fn, fn);
        queue = run.catch(() => {});
        return run;
    }

    /** Temp file + rename, so a crash mid-write cannot truncate the store.
     * The fixed name is deliberate: writes are already serialized per store
     * instance, so there is no concurrent writer to collide with, and reusing
     * one path means a crash between writeFile and rename leaves at most one
     * orphaned tmp file, self-cleaned by the next flush. */
    async function flush(): Promise<void> {
        const tmp = `${file}.tmp`;
        await fs.writeFile(tmp, JSON.stringify(state, null, 2));
        await fs.rename(tmp, file);
    }

    function mustGet(id: string): Person {
        const person = state.people.find(p => p.id === id);
        if (!person) throw new Error(`no such person: ${id}`);
        return person;
    }

    function mustGetReference(personId: string, refId: string): { person: Person; ref: Reference } {
        const person = mustGet(personId);
        const ref = person.references.find(r => r.id === refId);
        if (!ref) throw new Error(`no such reference: ${refId} on person ${personId}`);
        return { person, ref };
    }

    return {
        dataDir,
        list: () => state.people,
        get: (id) => state.people.find(p => p.id === id),

        addPerson: (name, ref) => serialize(async () => {
            const person: Person = {
                id: shortId('p'),
                name,
                createdAt: new Date().toISOString(),
                references: [{ ...ref, id: shortId('ref'), addedAt: new Date().toISOString() }],
            };
            state.people.push(person);
            await flush();
            return person;
        }),

        addReference: (personId, ref) => serialize(async () => {
            const person = mustGet(personId);
            person.references.push({
                ...ref, id: shortId('ref'), addedAt: new Date().toISOString(),
            });
            await flush();
            return person;
        }),

        deletePerson: (id) => serialize(async () => {
            state.people = state.people.filter(p => p.id !== id);
            await fs.rm(path.join(dataDir, 'people', id), { recursive: true, force: true });
            await flush();
        }),

        deleteReference: (personId, refId) => serialize(async () => {
            const person = mustGet(personId);
            if (person.references.length <= 1) {
                throw new Error('cannot delete the last reference of a person; delete the person instead');
            }
            person.references = person.references.filter(r => r.id !== refId);
            await flush();
            return person;
        }),

        // These two exist for callers that need to correct a single
        // reference's stored fields in place -- a post-hoc file relabel
        // (routes/people.ts), a re-embed after a pipeline change
        // (people/reembed.ts) -- WITHOUT the two things replaceAll implies:
        // (1) replacing the whole `people` array wholesale, which is only
        // safe when the caller captured it from inside this store's own
        // serialized queue (replaceAll's callers historically captured it
        // via `store.list()` from *outside* that queue, which a concurrent
        // mutation could race and silently undo); and (2) stamping
        // `pipelineVersion` to current, which both call sites need to NOT
        // do when the correction is partial (see reembed.ts). Deliberately
        // narrow to named fields on one reference -- do not widen this into
        // a generic "persist arbitrary state" method, which would recreate
        // both hazards under a new name.
        updateReferencePaths: (personId, refId, image, thumb) => serialize(async () => {
            const { person, ref } = mustGetReference(personId, refId);
            ref.image = image;
            ref.thumb = thumb;
            await flush();
            return person;
        }),

        updateReferenceEmbedding: (personId, refId, embedding, detScore) => serialize(async () => {
            const { person, ref } = mustGetReference(personId, refId);
            ref.embedding = embedding;
            ref.detScore = detScore;
            await flush();
            return person;
        }),

        replaceAll: (people) => serialize(async () => {
            state = { pipelineVersion: PIPELINE_VERSION, people };
            await flush();
        }),
    };
}

import 'dotenv/config';
import path from 'path';
import fs from 'fs/promises';
import { createApp } from './app.js';
import { createMatcher } from './face/matcher.js';
import { MATCH_THRESHOLD } from './face/score.js';
import { createStore, PIPELINE_VERSION } from './people/store.js';
import { reembedIfStale } from './people/reembed.js';

const port = 3100;

// Models take ~5s to compile. Load before listening so the first request is not
// an outlier and a missing model pack fails loudly at boot.
console.log('Loading face models (this takes a few seconds) ...');
const started = Date.now();
const matcher = await createMatcher();
console.log(`Models ready in ${((Date.now() - started) / 1000).toFixed(1)}s`);
console.log(`Match threshold: ${MATCH_THRESHOLD} (provisional, set FACE_MATCH_THRESHOLD to change)`);

const dataDir = path.join(process.cwd(), 'data');
const store = await createStore(dataDir);

// Drives reembedIfStale's staleness check below. Read straight from the file
// rather than through the store (which always reports the in-memory
// PIPELINE_VERSION-stamped shape) so a genuinely stale on-disk version is
// observed. A missing/unreadable people.json means first run -- nothing to
// re-embed -- so default to the current PIPELINE_VERSION, which never looks
// stale.
let storedVersion = PIPELINE_VERSION;
try {
    const raw = await fs.readFile(path.join(dataDir, 'people.json'), 'utf8');
    storedVersion = JSON.parse(raw).pipelineVersion ?? PIPELINE_VERSION;
} catch {
    // First run: no people.json yet.
}

// Must run before app.listen(): reembedIfStale mutates the live Person
// objects that store.list() returns, which is only safe with no request able
// to interleave, and a user with stale embeddings should never be served a
// search against them.
const reembedded = await reembedIfStale(store, matcher, storedVersion);
if (reembedded > 0) {
    console.log(`Re-embedded ${reembedded} saved reference(s) after a pipeline change`);
}

createApp(matcher, store).listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});

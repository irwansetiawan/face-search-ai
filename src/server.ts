import 'dotenv/config';
import { createApp } from './app.js';
import { createMatcher } from './face/matcher.js';
import { MATCH_THRESHOLD } from './face/score.js';

const port = 3100;

// Models take ~5s to compile. Load before listening so the first request is not
// an outlier and a missing model pack fails loudly at boot.
console.log('Loading face models (this takes a few seconds) ...');
const started = Date.now();
const matcher = await createMatcher();
console.log(`Models ready in ${((Date.now() - started) / 1000).toFixed(1)}s`);
console.log(`Match threshold: ${MATCH_THRESHOLD} (provisional, set FACE_MATCH_THRESHOLD to change)`);

createApp(matcher).listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});

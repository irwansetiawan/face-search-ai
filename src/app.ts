import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import type { Matcher } from './face/matcher.js';
import type { Store } from './people/store.js';
import { probeHandler } from './routes/probe.js';
import { searchHandler } from './routes/search.js';
import { peopleRouter } from './routes/people.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function createApp(matcher: Matcher, store: Store): express.Express {
    const app = express();
    const upload = multer({ dest: 'uploads/' });

    app.use(express.static(path.join(__dirname, '/static')));
    // Serves saved-people artefacts (downscaled originals + avatar crops)
    // written under store.dataDir/people/<id>/ by the people routes below.
    // Deliberately mounted on the people/ subdirectory only, NOT store.dataDir
    // itself -- store.dataDir also holds people.json, which carries every
    // 512-float embedding. Serving the whole dataDir as static files would
    // make that fetchable at /people-files/people.json, defeating the point
    // of publicPerson() stripping embeddings out of every API response.
    app.use('/people-files/people', express.static(path.join(store.dataDir, 'people')));
    app.post('/probe', upload.fields([{ name: 'source', maxCount: 1 }]), probeHandler(matcher));
    app.post('/search', upload.fields([{ name: 'target', maxCount: 1 }]), searchHandler(matcher, store));
    app.use('/people', peopleRouter(store, matcher));

    return app;
}

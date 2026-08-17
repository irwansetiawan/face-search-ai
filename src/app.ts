import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import type { Matcher } from './face/matcher.js';
import { probeHandler } from './routes/probe.js';
import { searchHandler } from './routes/search.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function createApp(matcher: Matcher): express.Express {
    const app = express();
    const upload = multer({ dest: 'uploads/' });

    app.use(express.static(path.join(__dirname, '/static')));
    app.post('/probe', upload.fields([{ name: 'source', maxCount: 1 }]), probeHandler(matcher));
    app.post('/search', upload.fields([{ name: 'target', maxCount: 1 }]), searchHandler(matcher));

    return app;
}

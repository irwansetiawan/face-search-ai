/**
 * Downloads the buffalo_l model pack. Not a postinstall hook: a silent 275 MB
 * download on `npm install` is unwelcome, and only two of the five files matter.
 */
import fs from 'fs';
import path from 'path';
import { pipeline } from 'stream/promises';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, '..', 'models', 'buffalo_l');
const URL_ = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip';
const REQUIRED = ['det_10g.onnx', 'w600k_r50.onnx'];

if (REQUIRED.every(f => fs.existsSync(path.join(MODEL_DIR, f)))) {
    console.log(`Models already present in ${MODEL_DIR}`);
    process.exit(0);
}

fs.mkdirSync(MODEL_DIR, { recursive: true });
const zipPath = path.join(MODEL_DIR, '..', 'buffalo_l.zip');

console.log(`Downloading buffalo_l (275 MB) from ${URL_} ...`);
const res = await fetch(URL_);
if (!res.ok) throw new Error(`Download failed: HTTP ${res.status}`);
await pipeline(res.body, fs.createWriteStream(zipPath));

console.log('Extracting ...');
execFileSync('unzip', ['-o', '-q', zipPath, '-d', MODEL_DIR]);
fs.unlinkSync(zipPath);

const missing = REQUIRED.filter(f => !fs.existsSync(path.join(MODEL_DIR, f)));
if (missing.length) throw new Error(`Extraction incomplete, missing: ${missing.join(', ')}`);
console.log(`Models ready in ${MODEL_DIR}`);

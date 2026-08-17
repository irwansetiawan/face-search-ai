import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** Repo-root-relative, resolved from dist/face/ at runtime. */
export const MODEL_DIR = path.join(__dirname, '..', '..', 'models', 'buffalo_l');
export const DET_MODEL = path.join(MODEL_DIR, 'det_10g.onnx');
export const REC_MODEL = path.join(MODEL_DIR, 'w600k_r50.onnx');

export function modelsPresent(det = DET_MODEL, rec = REC_MODEL): boolean {
    return fs.existsSync(det) && fs.existsSync(rec);
}

/** Fail with an actionable message rather than an ORT file-not-found error. */
export function assertModelsPresent(det = DET_MODEL, rec = REC_MODEL): void {
    if (!modelsPresent(det, rec)) {
        throw new Error(
            `Face models not found in ${MODEL_DIR}.\n` +
            `Run: npm run setup:models`
        );
    }
}

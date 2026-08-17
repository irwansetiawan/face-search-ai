import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** Repo-root-relative, resolved from dist/face/ at runtime. */
export const MODEL_DIR = path.join(__dirname, '..', '..', 'models', 'buffalo_l');
export const DET_MODEL = path.join(MODEL_DIR, 'det_10g.onnx');
export const REC_MODEL = path.join(MODEL_DIR, 'w600k_r50.onnx');

export function modelsPresent(): boolean {
    return fs.existsSync(DET_MODEL) && fs.existsSync(REC_MODEL);
}

/** Fail with an actionable message rather than an ORT file-not-found error. */
export function assertModelsPresent(): void {
    if (!modelsPresent()) {
        throw new Error(
            `Face models not found in ${MODEL_DIR}.\n` +
            `Run: npm run setup:models`
        );
    }
}

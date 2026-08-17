import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import { DET_MODEL, REC_MODEL, modelsPresent, assertModelsPresent } from './models.js';

test('model paths point at the two required onnx files', () => {
    assert.match(DET_MODEL, /buffalo_l\/det_10g\.onnx$/);
    assert.match(REC_MODEL, /buffalo_l\/w600k_r50\.onnx$/);
});

test('modelsPresent reflects what is on disk', () => {
    assert.equal(modelsPresent(), fs.existsSync(DET_MODEL) && fs.existsSync(REC_MODEL));
});

test('assertModelsPresent passes once models are downloaded', () => {
    assert.doesNotThrow(() => assertModelsPresent());
});

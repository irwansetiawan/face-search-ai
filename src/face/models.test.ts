import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import { DET_MODEL, REC_MODEL, modelsPresent, assertModelsPresent } from './models.js';

test('model paths point at the two required onnx files', () => {
    assert.match(DET_MODEL, /buffalo_l\/det_10g\.onnx$/);
    assert.match(REC_MODEL, /buffalo_l\/w600k_r50\.onnx$/);
});

test('modelsPresent returns true when both files are present', () => {
    assert.equal(modelsPresent(DET_MODEL, REC_MODEL), true);
});

test('modelsPresent returns false when detector is missing', () => {
    assert.equal(modelsPresent('/nonexistent/det_10g.onnx', REC_MODEL), false);
});

test('modelsPresent returns false when recognizer is missing', () => {
    assert.equal(modelsPresent(DET_MODEL, '/nonexistent/w600k_r50.onnx'), false);
});

test('modelsPresent returns false when both files are missing', () => {
    assert.equal(modelsPresent('/nonexistent/det_10g.onnx', '/nonexistent/w600k_r50.onnx'), false);
});

test('assertModelsPresent passes when both models are present', () => {
    assert.doesNotThrow(() => assertModelsPresent(DET_MODEL, REC_MODEL));
});

test('assertModelsPresent throws when models are missing', () => {
    try {
        assertModelsPresent('/nonexistent/det_10g.onnx', '/nonexistent/w600k_r50.onnx');
        assert.fail('Expected assertModelsPresent to throw');
    } catch (err) {
        const error = err as Error;
        assert.match(error.message, /Face models not found/);
        assert.match(error.message, /npm run setup:models/);
    }
});

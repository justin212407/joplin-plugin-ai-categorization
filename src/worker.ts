import { pipeline, env } from '@xenova/transformers';

const workerBase = self.location.href.replace(/[^/]*$/, '');
const wasmBase = `${workerBase}onnx-dist/`;

env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.wasmPaths = wasmBase;

let embedder: any = null;

self.onmessage = async (event) => {
    const { type, text, noteId } = event.data;

    if (type === 'load') {
        try {
            const loadStart = Date.now();
            embedder = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5');
            await embedder('warmup', { pooling: 'mean', normalize: true });
            self.postMessage({ type: 'loaded', loadMs: Date.now() - loadStart });
        } catch (err) {
            self.postMessage({ type: 'error', error: `Worker load failed: ${String(err)}` });
        }
    }

    if (type === 'embed') {
        if (!embedder) {
            self.postMessage({ type: 'error', noteId, error: 'Model not loaded' });
            return;
        }
        try {
            const start = Date.now();
            const output = await embedder(text.slice(0, 500), { pooling: 'mean', normalize: true });
            self.postMessage({
                type: 'embedding',
                noteId,
                embedding: Array.from(output.data),
                inferenceMs: Date.now() - start,
                dims: output.data.length
            });
        } catch (err) {
            self.postMessage({ type: 'error', noteId, error: `Worker embedding failed: ${String(err)}` });
        }
    }
};

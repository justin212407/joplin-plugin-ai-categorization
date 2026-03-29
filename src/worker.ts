import { pipeline, env } from '@xenova/transformers';

env.backends.onnx.wasm.wasmPaths = './onnx-dist/';

let embedder: any = null;

self.onmessage = async (event) => {
    const { type, text, noteId } = event.data;

    if (type === 'load') {
        const loadStart = Date.now();
        embedder = await pipeline(
            'feature-extraction',
            'Xenova/all-MiniLM-L6-v2'
        );
        // warmup
        await embedder('warmup', { pooling: 'mean', normalize: true });
        self.postMessage({ 
            type: 'loaded', 
            loadMs: Date.now() - loadStart 
        });
    }

    if (type === 'embed') {
        if (!embedder) {
            self.postMessage({ type: 'error', noteId, error: 'Model not loaded' });
            return;
        }
        const start = Date.now();
        const output = await embedder(
            text.slice(0, 500), 
            { pooling: 'mean', normalize: true }
        );
        self.postMessage({
            type: 'embedding',
            noteId,
            embedding: Array.from(output.data),
            inferenceMs: Date.now() - start,
            dims: output.data.length
        });
    }
};

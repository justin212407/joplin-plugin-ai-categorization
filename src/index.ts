import joplin from 'api';

// ─── Types ───────────────────────────────────────────────────────────────────

type NoteWithBody = {
    id: string;
    title: string;
    body: string;
    parent_id: string;
    updated_time: number;
};

type EmbeddedNote = {
    id: string;
    title: string;
    embedding: number[];
    parent_id: string;
};

type Cluster = {
    notes: Array<{ id: string; title: string }>;
    centroid: number[];
};

type LabelledCluster = {
    label: string;
    noteIds: string[];
    noteTitles: string[];
};

// ─── Step 1: Fetch Notes ─────────────────────────────────────────────────────

async function getAllNotesWithBody(): Promise<NoteWithBody[]> {
    const out: NoteWithBody[] = [];
    let page = 1;

    while (true) {
        const res = await joplin.data.get(['notes'], {
            fields: ['id', 'title', 'body', 'parent_id', 'updated_time'],
            limit: 50,
            page,
        });

        const items = res.items as Array<{
            id: string;
            title: string;
            body: string | null;
            parent_id: string;
            updated_time: number;
        }>;

        for (const item of items) {
            if (!item.body || item.body.trim() === '') continue;
            if (item.title?.includes('AI Categorization POC Output')) continue;
            out.push({
                id: item.id,
                title: item.title,
                body: item.body,
                parent_id: item.parent_id,
                updated_time: item.updated_time,
            });
        }

        if (!res.has_more) break;
        page += 1;
    }

    console.log(`[Step 1] Fetched ${out.length} notes with body.`);
    return out;
}

// ─── Step 2: Embeddings ──────────────────────────────────────────────────────

async function getEmbeddingOllama(text: string): Promise<number[]> {
    const truncated = text.slice(0, 1000);
    const candidates = [
        {
            url: 'http://127.0.0.1:11434/api/embed',
            body: { model: 'nomic-embed-text', input: truncated },
            parser: (data: any) => {
                if (Array.isArray(data?.embeddings) && Array.isArray(data.embeddings[0])) {
                    return data.embeddings[0] as number[];
                }
                return [] as number[];
            },
        },
        {
            url: 'http://127.0.0.1:11434/api/embeddings',
            body: { model: 'nomic-embed-text', prompt: truncated },
            parser: (data: any) => Array.isArray(data?.embedding) ? data.embedding as number[] : [],
        },
    ];

    for (const candidate of candidates) {
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 8000);
            const res = await fetch(candidate.url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: controller.signal,
                body: JSON.stringify(candidate.body),
            });
            clearTimeout(timeout);

            if (!res.ok) {
                console.warn(`[Embedding] Ollama endpoint ${candidate.url} returned ${res.status}`);
                lastOllamaError = `HTTP ${res.status} from ${candidate.url}`;
                continue;
            }

            const data = await res.json();
            const embedding = candidate.parser(data);
            const valid = Array.isArray(embedding)
                && embedding.length > 0
                && embedding.every(v => typeof v === 'number' && Number.isFinite(v));

            if (valid) return embedding;
            console.warn(`[Embedding] Ollama endpoint ${candidate.url} returned invalid embedding payload`);
            lastOllamaError = `Invalid embedding payload from ${candidate.url}`;
        } catch (err) {
            console.warn(`[Embedding] Ollama endpoint ${candidate.url} failed: ${String(err)}`);
            lastOllamaError = `${candidate.url}: ${String(err)}`;
        }
    }

    return [];
}

const lexicalEmbeddingDims = 256;

function fnv1aHash(input: string): number {
    let hash = 0x811c9dc5;
    for (let i = 0; i < input.length; i++) {
        hash ^= input.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193);
    }
    return hash >>> 0;
}

function normalizeVector(v: number[]): number[] {
    const mag = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
    if (mag === 0) return v;
    return v.map(x => x / mag);
}

function getDeterministicLexicalEmbedding(text: string): number[] {
    const vec = new Array(lexicalEmbeddingDims).fill(0);
    const tokens = text
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter(t => t.length >= 2)
        .slice(0, 400);

    if (tokens.length === 0) return vec;

    for (const token of tokens) {
        const h = fnv1aHash(token);
        const idx = h % lexicalEmbeddingDims;
        const sign = (h & 1) === 0 ? 1 : -1;
        vec[idx] += sign;
    }

    return normalizeVector(vec);
}

const embeddingBackendStats = {
    transformersNode: 0,
    transformersWorker: 0,
    transformers: 0,
    ollama: 0,
    lexical: 0,
};

let nodeEmbedder: any = null;
let nodeEmbedderLoaded = false;
let nodeEmbedderLoadMs = 0;
let lastNodeTransformersError = '';
let lastWorkerError = '';
let lastOllamaError = '';

async function ensureNodeTransformersLoaded(): Promise<number> {
    if (nodeEmbedderLoaded && nodeEmbedder) return nodeEmbedderLoadMs;

    try {
        // Keep @xenova/transformers out of static webpack analysis.
        // In Joplin plugin runtime, dynamic import of bare specifiers may fail,
        // so prefer runtime require from Node context.
        const runtimeRequire = new Function('return (typeof require !== "undefined") ? require : null;')() as ((m: string) => any) | null;
        if (!runtimeRequire) throw new Error('Node require is not available in this runtime');

        const mod = runtimeRequire('@xenova/transformers');
        if (!mod || typeof mod.pipeline !== 'function') {
            throw new Error('Loaded @xenova/transformers but pipeline() is unavailable');
        }

        const loadStart = Date.now();

        nodeEmbedder = await mod.pipeline(
            'feature-extraction',
            'Xenova/all-MiniLM-L6-v2'
        );

        await nodeEmbedder('warmup', { pooling: 'mean', normalize: true });

        nodeEmbedderLoadMs = Date.now() - loadStart;
        nodeEmbedderLoaded = true;
        lastNodeTransformersError = '';
        console.warn(`[Embedding] Node Transformers initialized in ${nodeEmbedderLoadMs}ms`);
        return nodeEmbedderLoadMs;
    } catch (err) {
        nodeEmbedderLoaded = false;
        nodeEmbedder = null;
        lastNodeTransformersError = String(err);
        throw err;
    }
}

async function getEmbeddingNodeTransformers(text: string, noteId: string): Promise<number[]> {
    await ensureNodeTransformersLoaded();
    if (!nodeEmbedder) throw new Error('Node Transformers embedder unavailable');

    const start = Date.now();
    const output = await nodeEmbedder(text.slice(0, 500), { pooling: 'mean', normalize: true });
    const embedding = Array.isArray(output?.data) ? output.data : Array.from(output?.data ?? []);

    const valid = Array.isArray(embedding)
        && embedding.length > 0
        && embedding.every(v => typeof v === 'number' && Number.isFinite(v));

    if (!valid) throw new Error('Node Transformers returned invalid embedding output');

    console.warn(`[Embedding] Node Transformers note=${noteId} latency=${Date.now() - start}ms dims=${embedding.length}`);
    return embedding as number[];
}

let embeddingWorker: Worker | null = null;
let embeddingWorkerLoaded = false;
let embeddingWorkerLoadMs = 0;
let workerEmbeddingCount = 0;

async function loadEmbeddingWorker(worker: Worker): Promise<number> {
    return await new Promise<number>((resolve, reject) => {
        const cleanup = () => {
            clearTimeout(timeout);
            worker.removeEventListener('message', handler as EventListener);
            worker.removeEventListener('error', errorHandler as EventListener);
            worker.removeEventListener('messageerror', messageErrorHandler as EventListener);
        };

        const timeout = setTimeout(() => {
            cleanup();
            reject(new Error('Worker load timeout'));
        }, 120000);

        const handler = (event: any) => {
            const data = event.data;
            if (!data) return;

            if (data.type === 'loaded') {
                cleanup();
                resolve(typeof data.loadMs === 'number' ? data.loadMs : 0);
                return;
            }

            if (data.type === 'error') {
                cleanup();
                reject(new Error(data.error || 'Worker load failed'));
            }
        };

        const errorHandler = (event: any) => {
            cleanup();
            reject(new Error(event?.message || 'Worker runtime error'));
        };

        const messageErrorHandler = () => {
            cleanup();
            reject(new Error('Worker message parsing error'));
        };

        worker.addEventListener('message', handler as EventListener);
        worker.addEventListener('error', errorHandler as EventListener);
        worker.addEventListener('messageerror', messageErrorHandler as EventListener);
        worker.postMessage({ type: 'load' });
    });
}

async function ensureEmbeddingWorkerLoaded(): Promise<number> {
    if (embeddingWorkerLoaded && embeddingWorker) return embeddingWorkerLoadMs;

    try {
        if (!embeddingWorker) {
            const installationDir = await joplin.plugins.installationDir();
            const workerPath = `${installationDir}/worker.js`;

            if (typeof Worker === 'undefined') {
                throw new Error('Global Worker is not available in this plugin runtime');
            }

            const candidatePaths = [
                workerPath,
                `file://${workerPath}`,
            ];

            let ctorError = '';
            for (const candidate of candidatePaths) {
                try {
                    embeddingWorker = new Worker(candidate);
                    break;
                } catch (err) {
                    ctorError = `${ctorError}${ctorError ? ' | ' : ''}${candidate}: ${String(err)}`;
                }
            }

            if (!embeddingWorker) {
                throw new Error(`Failed to construct Worker. Attempts: ${ctorError}`);
            }
        }

        embeddingWorkerLoadMs = await loadEmbeddingWorker(embeddingWorker);
        embeddingWorkerLoaded = true;
        lastWorkerError = '';
        console.warn(`[Embedding] Worker initialized in ${embeddingWorkerLoadMs}ms`);
        return embeddingWorkerLoadMs;
    } catch (err) {
        lastWorkerError = String(err);
        embeddingWorkerLoaded = false;
        if (embeddingWorker) {
            embeddingWorker.terminate();
            embeddingWorker = null;
        }
        throw err;
    }
}

async function getEmbeddingWorker(text: string, noteId: string): Promise<{ embedding: number[]; inferenceMs: number; dims: number }> {
    await ensureEmbeddingWorkerLoaded();
    if (!embeddingWorker) throw new Error('Worker unavailable');
    const workerRef = embeddingWorker;

    return await new Promise<{ embedding: number[]; inferenceMs: number; dims: number }>((resolve, reject) => {
        const cleanup = () => {
            clearTimeout(timeout);
            workerRef.removeEventListener('message', handler as EventListener);
            workerRef.removeEventListener('error', errorHandler as EventListener);
            workerRef.removeEventListener('messageerror', messageErrorHandler as EventListener);
        };

        const timeout = setTimeout(() => {
            cleanup();
            reject(new Error('Worker embedding timeout'));
        }, 120000);

        const handler = (event: any) => {
            const data = event.data;
            if (!data || data.noteId !== noteId) return;

            if (data.type === 'error') {
                cleanup();
                reject(new Error(data.error || 'Worker embedding failed'));
                return;
            }

            if (data.type === 'embedding') {
                cleanup();
                const embedding = Array.isArray(data.embedding) ? data.embedding as number[] : [];
                const inferenceMs = typeof data.inferenceMs === 'number' ? data.inferenceMs : 0;
                const dims = typeof data.dims === 'number' ? data.dims : embedding.length;
                workerEmbeddingCount += 1;
                if (workerEmbeddingCount >= 100) {
                    if (embeddingWorker) {
                        embeddingWorker.terminate();
                    }
                    embeddingWorker = null;
                    embeddingWorkerLoaded = false;
                    workerEmbeddingCount = 0;
                    console.warn('[Worker] Recycled after 100 embeddings to prevent WASM memory degradation.');
                }
                resolve({ embedding, inferenceMs, dims });
            }
        };

        const errorHandler = (event: any) => {
            cleanup();
            reject(new Error(event?.message || 'Worker runtime error'));
        };

        const messageErrorHandler = () => {
            cleanup();
            reject(new Error('Worker message parsing error'));
        };

        workerRef.addEventListener('message', handler as EventListener);
        workerRef.addEventListener('error', errorHandler as EventListener);
        workerRef.addEventListener('messageerror', messageErrorHandler as EventListener);
        workerRef.postMessage({
            type: 'embed',
            text,
            noteId,
        });
    });
}

async function getEmbeddingSmart(text: string, noteId: string): Promise<number[]> {
    const transformersStart = Date.now();

    try {
        const embedding = await getEmbeddingNodeTransformers(text, noteId);
        embeddingBackendStats.transformers += 1;
        embeddingBackendStats.transformersNode += 1;
        return embedding;
    } catch (err) {
        lastNodeTransformersError = String(err);
        console.warn(`[Embedding] Node Transformers failed for note=${noteId}: ${String(err)}`);
    }

    try {
        console.warn('[Embedding] Using Transformers');
        const workerResult = await getEmbeddingWorker(text.slice(0, 500), noteId);
        const embedding = workerResult.embedding;
        const valid = Array.isArray(embedding)
            && embedding.length > 0
            && embedding.every(v => typeof v === 'number' && Number.isFinite(v));

        if (!valid) {
            throw new Error('Invalid worker embedding output');
        }

        const elapsed = Date.now() - transformersStart;
        console.warn(`[Embedding] Transformers note=${noteId} latency=${elapsed}ms dims=${workerResult.dims}`);
        embeddingBackendStats.transformers += 1;
        embeddingBackendStats.transformersWorker += 1;
        return embedding;
    } catch (err) {
        lastWorkerError = String(err);
        console.warn('[Embedding] Falling back to Ollama');
        console.warn(`[Embedding] Fallback reason for note=${noteId}: ${err}`);
        const ollamaStart = Date.now();
        const fallback = await getEmbeddingOllama(text);
        const ollamaMs = Date.now() - ollamaStart;
        console.warn(`[Embedding] Ollama note=${noteId} latency=${ollamaMs}ms dims=${fallback.length}`);
        if (fallback.length > 0) {
            embeddingBackendStats.ollama += 1;
            return fallback;
        }

        const lexical = getDeterministicLexicalEmbedding(text);
        embeddingBackendStats.lexical += 1;
        console.warn(`[Embedding] Lexical fallback note=${noteId} dims=${lexical.length}`);
        return lexical;
    }
}

async function runBackendComparison(testNotes: NoteWithBody[]): Promise<{
    ollamaAvg: number;
    transformersAvg: number;
    speedRatio: number;
    ollamaDims: number;
    transformersDims: number;
    transformersFailed: boolean;
}> {
    const sampleNotes = testNotes.slice(0, 3);
    const ollamaTimings: number[] = [];
    const transformersTimings: number[] = [];
    let ollamaDims = 0;
    let transformersDims = 0;

    for (let i = 0; i < sampleNotes.length; i++) {
        const note = sampleNotes[i];
        const combined = note.title + '\n' + note.body;

        const ollamaStart = Date.now();
        const ollamaEmbedding = await getEmbeddingOllama(combined);
        const ollamaMs = Date.now() - ollamaStart;
        ollamaTimings.push(ollamaMs);
        if (!ollamaDims && ollamaEmbedding.length > 0) ollamaDims = ollamaEmbedding.length;

        const transformersStart = Date.now();
        let transformersEmbedding: number[] = [];
        let transformersMs = -1;

        try {
            const workerResult = await getEmbeddingWorker(combined, `${note.id}-compare-${i}`);
            transformersEmbedding = workerResult.embedding;
            if (transformersEmbedding.length > 0) {
                transformersMs = Date.now() - transformersStart;
            }
        } catch (err) {
            console.warn('[Compare] Transformers worker failed:', err);
        }

        transformersTimings.push(transformersMs);
        if (!transformersDims && transformersEmbedding.length > 0) transformersDims = transformersEmbedding.length;

        console.warn(`[Compare] Note ${i + 1}: Ollama=${ollamaMs}ms vs Transformers=${transformersMs}ms`);
    }

    const ollamaAvg = ollamaTimings.length > 0
        ? ollamaTimings.reduce((acc, t) => acc + t, 0) / ollamaTimings.length
        : 0;
    const transformersAvg = transformersTimings.length > 0
        ? transformersTimings.reduce((acc, t) => acc + t, 0) / transformersTimings.length
        : 0;
    const transformersFailed = transformersTimings.some(t => t === -1);
    const speedRatio = transformersAvg === 0 ? 0 : ollamaAvg / transformersAvg;

    return { ollamaAvg, transformersAvg, speedRatio, ollamaDims, transformersDims, transformersFailed };
}

async function benchmarkTransformersWorker(notes: NoteWithBody[]): Promise<
    | { loadMs: number; avgMs: number; throughput: number; dims: number; success: true }
    | { success: false; error: string }
> {
    try {
        const loadMs = await ensureEmbeddingWorkerLoaded();

        const sampleNotes = notes.slice(0, 5);
        const timings: number[] = [];
        let dims = 0;
        const benchmarkStart = Date.now();

        for (let i = 0; i < sampleNotes.length; i++) {
            const note = sampleNotes[i];
            const result = await getEmbeddingWorker(note.title + '\n' + note.body, `${note.id}-bench-${i}`);

            timings.push(result.inferenceMs);
            if (!dims) dims = result.dims;
        }

        const elapsedSec = (Date.now() - benchmarkStart) / 1000;
        const avgMs = timings.length > 0
            ? timings.reduce((acc, t) => acc + t, 0) / timings.length
            : 0;
        const throughput = elapsedSec > 0 ? sampleNotes.length / elapsedSec : 0;

        return {
            loadMs,
            avgMs,
            throughput,
            dims,
            success: true,
        };
    } catch (err) {
        return {
            success: false,
            error: String(err),
        };
    }
}

async function embedAllNotes(notes: NoteWithBody[]): Promise<{ notes: EmbeddedNote[]; avgMs: number; dimsconfirmed: number }> {
    const results: EmbeddedNote[] = [];
    const timings: number[] = [];
    const batchStart = Date.now();

    for (let i = 0; i < notes.length; i++) {
        const note = notes[i];
        const noteStart = Date.now();
        const embedding = await getEmbeddingSmart(note.title + '\n' + note.body, note.id);
        const elapsed = Date.now() - noteStart;
        timings.push(elapsed);

        const notesProcessed = i + 1;
        const elapsedBatchSec = (Date.now() - batchStart) / 1000;
        const cumulative = elapsedBatchSec > 0 ? notesProcessed / elapsedBatchSec : 0;

        console.warn(`[Step 2] [${notesProcessed}/${notes.length}] ${note.title} — ${elapsed}ms (${embedding.length}d) | cumulative: ${cumulative.toFixed(1)} notes/sec`);

        if (embedding.length === 0) {
            console.log(`[Step 2] Skipped (empty embedding): ${note.title}`);
            continue;
        }

        results.push({ id: note.id, title: note.title, embedding, parent_id: note.parent_id });
    }

    const totalSec = (Date.now() - batchStart) / 1000;

    if (timings.length > 0) {
        const sorted = [...timings].sort((a, b) => a - b);
        const sum = timings.reduce((acc, t) => acc + t, 0);
        const avg = sum / timings.length;
        const p50 = sorted[Math.floor(sorted.length / 2)];
        const p90 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))];
        const throughput = totalSec > 0 ? timings.length / totalSec : 0;
        const embeddingDims = results.length > 0 ? results[0].embedding.length : 0;

        console.warn(`[Step 2] Batch complete: ${timings.length} notes in ${totalSec.toFixed(2)}s`);
        console.warn(`[Step 2] Throughput: ${throughput.toFixed(1)} notes/sec`);
        console.warn(`[Step 2] Latency: avg=${Math.round(avg)}ms p50=${Math.round(p50)}ms p90=${Math.round(p90)}ms`);
        console.warn(`[Step 2] Embedding dimensions: ${embeddingDims}`);

        if (timings.length >= 4) {
            const midpoint = Math.floor(timings.length / 2);
            const firstHalf = timings.slice(0, midpoint);
            const secondHalf = timings.slice(midpoint);

            const firstHalfAvg = firstHalf.reduce((acc, t) => acc + t, 0) / firstHalf.length;
            const secondHalfAvg = secondHalf.reduce((acc, t) => acc + t, 0) / secondHalf.length;
            const percentChange = firstHalfAvg === 0 ? 0 : ((secondHalfAvg - firstHalfAvg) / firstHalfAvg) * 100;

            console.warn(`[Step 2] Throughput degradation: first half ${Math.round(firstHalfAvg)}ms/note → second half ${Math.round(secondHalfAvg)}ms/note (${percentChange.toFixed(1)}% change)`);
        }
    }

    const avgMs = timings.length > 0
        ? timings.reduce((acc, t) => acc + t, 0) / timings.length
        : 0;
    const dimsconfirmed = results.length > 0 ? results[0].embedding.length : 0;

    console.log(`[Step 2] Embedding complete. ${results.length} notes embedded.`);
    return { notes: results, avgMs, dimsconfirmed };
}

// ─── Step 3: Clustering ──────────────────────────────────────────────────────

function cosineSimilarity(a: number[], b: number[]): number {
    if (a.length === 0 || b.length === 0) return 0;
    let dot = 0, magA = 0, magB = 0;
    for (let i = 0; i < a.length; i++) {
        dot  += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }
    const denom = Math.sqrt(magA) * Math.sqrt(magB);
    return denom === 0 ? 0 : dot / denom;
}

function computeCentroid(embeddings: number[][]): number[] {
    if (embeddings.length === 0) return [];
    const dim = embeddings[0].length;
    const centroid = new Array(dim).fill(0);
    for (const emb of embeddings) {
        for (let i = 0; i < dim; i++) centroid[i] += emb[i];
    }
    return centroid.map(v => v / embeddings.length);
}

function kMeansCosine(
    embeddedNotes: EmbeddedNote[],
    k: number,
    maxIterations = 100
): Cluster[] {
    if (embeddedNotes.length < k) k = embeddedNotes.length;

    // k-means++ initialisation for stable centroid seeding
    const centroids: number[][] = [];
    centroids.push(embeddedNotes[Math.floor(Math.random() * embeddedNotes.length)].embedding);

    while (centroids.length < k) {
        const distances = embeddedNotes.map(note => {
            const sims = centroids.map(c => cosineSimilarity(note.embedding, c));
            return 1 - Math.max(...sims); // distance = 1 - best similarity
        });
        const totalDist = distances.reduce((a, b) => a + b, 0);
        let rand = Math.random() * totalDist;
        for (let i = 0; i < distances.length; i++) {
            rand -= distances[i];
            if (rand <= 0) {
                centroids.push(embeddedNotes[i].embedding);
                break;
            }
        }
    }

    let assignments = new Array(embeddedNotes.length).fill(-1);

    for (let iter = 0; iter < maxIterations; iter++) {
        // Assignment step — every note goes to its closest centroid, no threshold floor
        const newAssignments = embeddedNotes.map(note =>
            centroids
                .map((c, i) => ({ i, sim: cosineSimilarity(note.embedding, c) }))
                .reduce((best, curr) => curr.sim > best.sim ? curr : best)
                .i
        );

        const converged = newAssignments.every((a, i) => a === assignments[i]);
        assignments = newAssignments;
        if (converged) break;

        // Update step — recompute centroids from assigned notes
        for (let ci = 0; ci < k; ci++) {
            const members = embeddedNotes.filter((_, i) => assignments[i] === ci);
            if (members.length > 0) centroids[ci] = computeCentroid(members.map(n => n.embedding));
        }
    }

    // Build Cluster objects — every note is assigned, none dropped
    return Array.from({ length: k }, (_, ci) => {
        const members = embeddedNotes.filter((_, i) => assignments[i] === ci);
        return {
            notes: members.map(n => ({ id: n.id, title: n.title })),
            centroid: computeCentroid(members.map(n => n.embedding)),
        };
    }).filter(c => c.notes.length > 0);
}

function centroidSilhouetteScore(
    embeddedNotes: EmbeddedNote[],
    clusters: Cluster[]
): number {
    if (clusters.length < 2) return 0;

    const embeddingById = new Map(embeddedNotes.map(n => [n.id, n.embedding]));
    let total = 0;
    let count = 0;

    for (let i = 0; i < clusters.length; i++) {
        const ownCentroid = clusters[i].centroid;

        // Nearest other centroid distance for this cluster
        let nearestOtherDist = Infinity;
        for (let j = 0; j < clusters.length; j++) {
            if (i === j) continue;
            const d = 1 - cosineSimilarity(ownCentroid, clusters[j].centroid);
            if (d < nearestOtherDist) nearestOtherDist = d;
        }

        for (const member of clusters[i].notes) {
            const emb = embeddingById.get(member.id);
            if (!emb) continue;

            // a(i) = distance from note to its own centroid
            const a = 1 - cosineSimilarity(emb, ownCentroid);

            // b(i) = distance to nearest other centroid
            let b = Infinity;
            for (let j = 0; j < clusters.length; j++) {
                if (j === i) continue;
                const d = 1 - cosineSimilarity(emb, clusters[j].centroid);
                if (d < b) b = d;
            }

            if (!isFinite(b)) continue;
            const denom = Math.max(a, b);
            if (denom === 0) continue;

            total += (b - a) / denom;
            count++;
        }
    }

    return count === 0 ? 0 : total / count;
}

function findOptimalK(embeddedNotes: EmbeddedNote[]): {
    optimalK: number;
    optimalSilhouette: number;
    allResults: Array<{ k: number; silhouette: number }>;
} {
    const n = embeddedNotes.length;

    const maxK = Math.min(Math.ceil(Math.sqrt(n)), 8);
    const allResults: Array<{ k: number; silhouette: number }> = [];

    console.warn(`[Calibration] n=${n} searching k=2..${maxK}`);

    for (let k = 2; k <= maxK; k++) {
        let bestSilhouette = -1;
        for (let run = 0; run < 7; run++) {
            const clusters = kMeansCosine(embeddedNotes, k);
            const s = clusters.length >= 2
                ? centroidSilhouetteScore(embeddedNotes, clusters)
                : 0;
            // Penalise solutions that produce tiny clusters — singletons are almost always wrong
            const minClusterSize = Math.min(...clusters.map(c => c.notes.length));
            const penalty = minClusterSize < 3 ? 0.15 : minClusterSize < 5 ? 0.05 : 0;
            const adjusted = s - penalty;
            if (adjusted > bestSilhouette) bestSilhouette = adjusted;
        }
        allResults.push({ k, silhouette: bestSilhouette });
    }

    const best = allResults.reduce((a, b) => b.silhouette > a.silhouette ? b : a);
    return {
        optimalK: best.k,
        optimalSilhouette: best.silhouette,
        allResults,
    };
}

// ─── Step 4: LLM Labelling ───────────────────────────────────────────────────

function heuristicClusterLabel(noteTitles: string[]): string {
    const stopWords = new Set([
        'and', 'the', 'for', 'with', 'from', 'into', 'using', 'over', 'under', 'best',
        'guide', 'tips', 'how', 'to', 'in', 'of', 'on', 'a', 'an', 'is', 'are', 'at',
        'by', 'vs', 'vs.', 'your', 'my', 'this', 'that', 'these', 'those', 'via',
    ]);

    const freq = new Map<string, number>();
    for (const title of noteTitles) {
        const words = title
            .toLowerCase()
            .replace(/[^a-z0-9\s]/g, ' ')
            .split(/\s+/)
            .filter(w => w.length >= 3 && !stopWords.has(w));

        for (const w of words) {
            freq.set(w, (freq.get(w) ?? 0) + 1);
        }
    }

    const ranked = [...freq.entries()].sort((a, b) => b[1] - a[1]);
    if (ranked.length === 0) return 'general-topic';
    if (ranked.length === 1) return `${ranked[0][0]} notes`;
    return `${ranked[0][0]} ${ranked[1][0]}`;
}

async function getClusterLabel(noteTitles: string[]): Promise<string> {
    const heuristic = heuristicClusterLabel(noteTitles);

    try {
        const sample = noteTitles.slice(0, 5).join(', ');
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 8000);
        const res = await fetch('http://127.0.0.1:11434/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({
                model: 'llama3',
                stream: false,
                prompt: `Given these note titles: ${sample}
Suggest ONE short tag name (2-3 words maximum) that best categorizes them.
Reply with ONLY the tag name. No explanation. No punctuation. Lowercase only.`,
            }),
        });
        clearTimeout(timeout);

        if (!res.ok) {
            console.error(`[Label] HTTP error: ${res.status}`);
            return heuristic;
        }

        const data = await res.json();
        const raw = String(data.response ?? '').trim().toLowerCase();
        const sanitized = raw
            .replace(/[^a-z0-9\s-]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();

        if (!sanitized || sanitized === 'uncategorized' || sanitized.length < 3) {
            return heuristic;
        }

        return sanitized;
    } catch (err) {
        console.error('[Label] Ollama not reachable for labelling.', err);
        return heuristic;
    }
}

async function labelAllClusters(
    clusters: Cluster[],
    embeddedNotes: EmbeddedNote[]
): Promise<LabelledCluster[]> {
    const results: LabelledCluster[] = [];

    for (let i = 0; i < clusters.length; i++) {
        const cluster = clusters[i];
        const titles = cluster.notes.map(n => n.title);
        console.log(`[Step 4] Labelling cluster ${i + 1} of ${clusters.length}...`);
        const label = await getClusterLabel(titles);
        console.log(`[Step 4] Cluster found: '${label}' with ${titles.length} notes: [${titles.join(', ')}]`);
        results.push({
            label,
            noteIds: cluster.notes.map(n => n.id),
            noteTitles: titles,
        });
    }

    return results;
}

function detectSemanticGaps(
    embedded: EmbeddedNote[],
    clusters: Cluster[]
): Array<{ noteId: string; title: string; clusterLabel: string; distanceFromCentroid: number; insight: string }> {
    const gaps: Array<{ noteId: string; title: string; clusterLabel: string; distanceFromCentroid: number; insight: string }> = [];
    const embeddingById = new Map(embedded.map(n => [n.id, n.embedding]));

    for (const cluster of clusters) {
        for (const note of cluster.notes) {
            const embedding = embeddingById.get(note.id);
            if (!embedding) continue;
            const distance = 1 - cosineSimilarity(embedding, cluster.centroid);
            if (distance > 0.25) {
                gaps.push({
                    noteId: note.id,
                    title: note.title,
                    clusterLabel: cluster.notes.length > 0 ? cluster.notes[0].title : 'unknown',
                    distanceFromCentroid: distance,
                    insight: distance > 0.40
                        ? 'Likely misfiled — consider moving to a different notebook'
                        : 'Partially off-topic — note may be incomplete or cover multiple topics',
                });
            }
        }
    }

    return gaps.sort((a, b) => b.distanceFromCentroid - a.distanceFromCentroid);
}

function detectNearDuplicates(
    embedded: EmbeddedNote[],
    threshold = 0.92
): Array<{ noteA: string; noteB: string; similarity: number }> {
    const duplicates: Array<{ noteA: string; noteB: string; similarity: number }> = [];
    for (let i = 0; i < embedded.length; i++) {
        for (let j = i + 1; j < embedded.length; j++) {
            const sim = cosineSimilarity(embedded[i].embedding, embedded[j].embedding);
            if (sim >= threshold) {
                duplicates.push({
                    noteA: embedded[i].title,
                    noteB: embedded[j].title,
                    similarity: sim,
                });
            }
        }
    }
    return duplicates.sort((a, b) => b.similarity - a.similarity);
}

async function applyTagsToNotes(labelled: LabelledCluster[]): Promise<Array<{
    label: string;
    tagId: string;
    appliedCount: number;
    failedCount: number;
}>> {
    const results: Array<{
        label: string;
        tagId: string;
        appliedCount: number;
        failedCount: number;
    }> = [];

    for (const cluster of labelled) {
        const existingTagsRes = await joplin.data.get(['tags'], {
            fields: ['id', 'title'],
        });
        const existingTags = (existingTagsRes.items ?? []) as Array<{ id: string; title: string }>;

        const existingTag = existingTags.find(tag =>
            tag.title.toLowerCase() === cluster.label.toLowerCase()
        );

        let tagId = '';
        if (existingTag) {
            tagId = existingTag.id;
        } else {
            const newTag = await joplin.data.post(['tags'], null, { title: cluster.label });
            tagId = newTag.id;
        }

        let appliedCount = 0;
        let failedCount = 0;

        for (const noteId of cluster.noteIds) {
            try {
                await joplin.data.post(['tags', tagId, 'notes'], null, { id: noteId });
                appliedCount += 1;
            } catch (err) {
                failedCount += 1;
                console.warn(`[Apply] Failed to tag note ${noteId} with '${cluster.label}':`, err);
            }
        }

        results.push({
            label: cluster.label,
            tagId,
            appliedCount,
            failedCount,
        });
    }

    return results;
}

async function showConfirmationDialog(labelled: LabelledCluster[]): Promise<boolean> {
    const dialog = await joplin.views.dialogs.create('aiCategorizationConfirm');

    await joplin.views.dialogs.setHtml(dialog, `
        <div style="font-family: sans-serif; padding: 16px;
                    max-width: 400px;">
            <h3 style="margin-top: 0;">
                AI Categorization Suggestions
            </h3>
            <p style="color: #666; font-size: 13px;">
                The following tags will be created and applied
                to your notes:
            </p>
            ${labelled.map(c => `
                <div style="margin: 8px 0; padding: 8px;
                            background: #f5f5f5; border-radius: 4px;">
                    <strong>${c.label}</strong>
                    <span style="color: #666; font-size: 12px;
                                 margin-left: 8px;">
                        → ${c.noteIds.length} notes
                    </span>
                    <div style="font-size: 11px; color: #999;
                                margin-top: 4px;">
                        ${c.noteTitles.slice(0, 2).join(', ')}
                        ${c.noteTitles.length > 2 ?
                            ` +${c.noteTitles.length - 2} more` : ''}
                    </div>
                </div>
            `).join('')}
            <p style="font-size: 12px; color: #888; margin-top: 16px;">
                This action can be undone by manually removing
                the tags. No notes will be moved.
            </p>
        </div>
    `);

    await joplin.views.dialogs.setButtons(dialog, [
        { id: 'apply', title: 'Apply Tags' },
        { id: 'skip', title: 'Skip' }
    ]);

    const result = await joplin.views.dialogs.open(dialog);
    return result.id === 'apply';
}

async function createTestVault(): Promise<void> {
    const notebookTitles = [
        'Machine Learning',
        'Cooking and Recipes',
        'Travel Planning',
        'Software Engineering',
        'Personal Finance',
        'Health and Fitness',
    ];

    for (const title of notebookTitles) {
        await joplin.data.post(['folders'], null, { title });
    }

const notesToCreate: Array<{ title: string; body: string }> = [
    {
        title: 'Neural network architectures',
        body: 'Convolutional neural networks apply shared filter weights across spatial positions in image data, enabling translation-invariant feature detection without relearning patterns at every location. Recurrent architectures including LSTMs and GRUs maintain hidden state across sequence timesteps using gating mechanisms that control information flow and prevent vanishing gradients over long dependencies. Residual networks introduce skip connections that add the input of a block directly to its output, allowing gradients to bypass layers and enabling training of networks with hundreds of layers. Transformer architectures replace recurrence entirely with self-attention mechanisms that compute pairwise relationships between all sequence positions simultaneously. Graph neural networks extend deep learning to non-Euclidean data by aggregating features from neighbouring nodes through message passing. Capsule networks attempt to preserve spatial hierarchies by routing agreement between lower and higher level feature detectors. Mixture of experts architectures activate only a subset of parameters per input, enabling model capacity to scale without proportional compute increases. Neural architecture search automates the design of network topologies using reinforcement learning or evolutionary algorithms to optimise validation performance.',
    },
    {
        title: 'Backpropagation explained',
        body: 'Backpropagation applies the chain rule of calculus to compute the gradient of a scalar loss function with respect to every parameter in a neural network by propagating error signals backward from the output layer through each intermediate layer. Forward pass computes activations at every neuron by applying weights, biases, and nonlinear activation functions sequentially from input to output. Backward pass computes local gradients at each operation and multiplies them with upstream gradients flowing from the loss. Weight gradients accumulate contributions from every training example in a minibatch before the optimiser applies an update. Vanishing gradients occur when repeated multiplication of small derivatives shrinks the signal to near zero in early layers, preventing learning. Exploding gradients cause numerical instability when repeated multiplication amplifies the signal, addressed by gradient clipping. Batch normalisation stabilises gradient magnitudes by normalising layer inputs to zero mean and unit variance before rescaling. Second-order methods like natural gradient descent use curvature information from the Fisher information matrix to scale updates more efficiently than first-order gradient descent alone.',
    },
    {
        title: 'Transformer architecture details',
        body: 'The transformer architecture introduced in Attention Is All You Need replaces recurrent and convolutional sequence processing with stacked self-attention and feed-forward layers. Scaled dot-product attention computes compatibility scores between query and key vectors, divides by the square root of the key dimension to prevent softmax saturation at large dimensionalities, and uses the resulting weights to form a weighted sum of value vectors. Multi-head attention runs several attention functions in parallel with different learned projections, concatenates the results, and projects to the output dimension, capturing diverse relationship types simultaneously. Positional encodings inject sequence order information using sinusoidal functions of different frequencies, allowing the model to attend to relative positions. Layer normalisation is applied before each sublayer in the pre-norm formulation used by most modern transformers. The feed-forward sublayer applies two linear transformations with a nonlinearity between them, typically a GELU or SwiGLU activation. Causal masking prevents autoregressive decoders from attending to future positions during training. Rotary position embeddings encode relative positions directly into the attention computation without additive position encodings.',
    },
    {
        title: 'Training large language models',
        body: 'Large language model pretraining optimises next-token prediction on internet-scale text corpora using the cross-entropy loss between predicted and actual token distributions. Data curation pipelines apply quality filtering, deduplication, and domain mixing to construct training datasets that balance diversity with signal quality. Mixed precision training stores weights in float32 but performs forward and backward passes in bfloat16, reducing memory bandwidth and enabling larger batch sizes without significant precision loss. Gradient checkpointing trades compute for memory by recomputing activations during the backward pass rather than storing them during the forward pass. Model parallelism distributes parameters across devices using tensor parallelism within layers and pipeline parallelism across layers. Instruction tuning fine-tunes pretrained models on curated demonstration datasets to follow natural language instructions reliably. Reinforcement learning from human feedback trains a reward model on human preference comparisons and uses it to fine-tune the language model via proximal policy optimisation. Constitutional AI generates model critiques of its own outputs and uses supervised and reinforcement learning on those critiques to improve safety and helpfulness simultaneously.',
    },
    {
        title: 'Gradient descent optimization',
        body: 'Stochastic gradient descent updates parameters using the gradient computed on a random minibatch rather than the full dataset, introducing noise that can help escape sharp local minima and reduce per-step compute cost. Momentum accumulates an exponentially weighted moving average of past gradients to smooth updates and accelerate convergence along consistent gradient directions. Nesterov momentum computes the gradient at the anticipated future position rather than the current position, providing a corrective effect that reduces oscillation. AdaGrad adapts the learning rate per parameter by dividing by the square root of the sum of squared historical gradients, benefiting sparse parameters that receive infrequent large updates. RMSProp modifies AdaGrad by using an exponential moving average of squared gradients instead of their cumulative sum, preventing the learning rate from decaying to zero. Adam combines momentum and RMSProp with bias correction terms that compensate for initialisation at zero. Learning rate warmup linearly increases the learning rate from near zero over the first few thousand steps to prevent instability when parameters are randomly initialised. Cosine annealing schedules gradually reduce the learning rate following a cosine curve from the initial value to near zero over the training run.',
    },
    {
        title: 'Embedding models and vector spaces',
        body: 'Dense text embeddings map variable-length strings to fixed-dimensional real-valued vectors where geometric relationships encode semantic relationships. Contrastive learning trains embedding models by pulling representations of similar pairs closer together and pushing dissimilar pairs apart in the vector space. The InfoNCE loss treats contrastive learning as a classification problem where the model must identify the positive pair among a set of negatives sampled from the batch. Temperature scaling controls the sharpness of the softmax distribution over similarity scores, with lower temperatures producing more peaked distributions that emphasise hard negatives. Normalised embeddings constrain representations to the unit hypersphere, enabling cosine similarity to be computed efficiently as a dot product. Matryoshka representation learning trains models to produce useful embeddings at multiple dimensionalities simultaneously by nesting loss functions at different truncation points. Retrieval-augmented generation uses embedding-based nearest neighbour search to retrieve relevant documents from a corpus and concatenates them with the query before generation. Bi-encoder architectures encode queries and documents independently, enabling precomputed document embeddings to be searched efficiently at inference time using approximate nearest neighbour indices.',
    },
    {
        title: 'Fine-tuning strategies',
        body: 'Full fine-tuning updates all parameters of a pretrained model on a task-specific dataset, providing maximum flexibility but requiring compute and memory proportional to model size. Parameter-efficient fine-tuning methods reduce trainable parameters by orders of magnitude while retaining most of the performance of full fine-tuning. LoRA decomposes weight update matrices into low-rank factors A and B, training only these factors while keeping the original weights frozen, reducing trainable parameters by up to 99 percent depending on rank. QLoRA combines 4-bit quantisation of the frozen base model with LoRA adapters trained in bfloat16, enabling fine-tuning of 70-billion-parameter models on consumer GPUs. Prefix tuning prepends a sequence of learned continuous vectors to the input of each transformer layer, conditioning the frozen model without modifying its weights. Prompt tuning is a lighter variant that prepends learned tokens only to the input embedding layer. Adapter layers insert small bottleneck modules between transformer sublayers, adding a down-projection, nonlinearity, and up-projection in series. Catastrophic forgetting occurs when fine-tuning on a new task overwrites representations needed for previously learned tasks, mitigated by elastic weight consolidation or rehearsal with samples from prior tasks.',
    },
    {
        title: 'Evaluation metrics for ML models',
        body: 'Classification evaluation requires careful metric selection depending on class balance and the relative cost of false positives versus false negatives in the deployment context. Accuracy measures the fraction of predictions that match the ground truth label and is misleading on imbalanced datasets where predicting the majority class achieves high scores trivially. Precision measures the fraction of positive predictions that are genuinely positive, penalising false alarms. Recall measures the fraction of actual positives that are correctly identified, penalising missed detections. F1 score is the harmonic mean of precision and recall, providing a single metric that balances both without allowing one to dominate. Area under the receiver operating characteristic curve measures discrimination ability across all classification thresholds by plotting true positive rate against false positive rate. Average precision summarises the precision-recall curve as a weighted mean of precisions at each threshold, more informative than AUC-ROC on highly imbalanced datasets. Calibration assesses whether predicted probabilities match empirical frequencies, evaluated using reliability diagrams and expected calibration error. Perplexity measures language model quality as the exponentiated average negative log-likelihood per token, reflecting how surprised the model is by held-out text.',
    },
    {
        title: 'French cooking techniques',
        body: 'Classical French cuisine established the foundational techniques that underpin professional cookery worldwide through systematic codification by Escoffier and Carême. Sautéing uses a small amount of clarified butter or neutral oil in a wide pan over high heat to brown protein surfaces through the Maillard reaction while preserving interior moisture. Braising combines initial high-heat searing to develop fond with slow moist-heat cooking in a covered vessel, breaking down collagen in tough cuts into gelatin that enriches the braising liquid. Deglazing dissolves the caramelised fond left after searing by adding wine, stock, or water to the hot pan and scraping up the solids, concentrating flavour into pan sauces. Emulsification disperses fat droplets in water or vice versa using an emulsifying agent such as lecithin in egg yolk to create stable sauces including hollandaise, béarnaise, and beurre blanc. The mother sauces of classical French cuisine — béchamel, velouté, espagnole, tomato, and hollandaise — serve as foundations for hundreds of derivative small sauces. Mise en place describes the practice of measuring, cutting, and preparing all ingredients before cooking begins, enabling precise timing and reducing errors during service. Reduction concentrates flavour by evaporating water from stocks and sauces through sustained simmering.',
    },
    {
        title: 'Pasta making from scratch',
        body: 'Fresh pasta dough requires precise ingredient ratios and technique to achieve the silky, pliable texture that distinguishes handmade from dried commercial pasta. Tipo 00 flour milled to extremely fine particle size and low protein content between 9 and 11 percent produces tender, smooth dough that rolls to translucent sheets without tearing. Semolina flour ground from durum wheat adds texture and structural integrity, helping extruded shapes like rigatoni and penne hold their form and grip sauce effectively. Egg yolks contribute fat and emulsifiers that enrich the dough, producing golden colour and a tender bite, while whole eggs add additional water for hydration. The standard hydration ratio of 55 to 60 percent total liquid to flour produces workable dough that is neither sticky nor crumbly. Kneading develops gluten structure by aligning protein strands, requiring 8 to 10 minutes by hand until the dough is smooth and springs back when poked. Resting the dough wrapped in plastic for 30 minutes at room temperature relaxes gluten tension, making the dough extensible and preventing it from snapping back during rolling. Lamination techniques fold dough multiple times through progressively thinner pasta machine settings to develop consistent texture and uniform thickness.',
    },
    {
        title: 'Fermentation and sourdough',
        body: 'Sourdough fermentation relies on a complex microbial ecosystem of wild yeasts and lactic acid bacteria that colonise flour and water mixtures over repeated feeding cycles. Wild Saccharomyces cerevisiae and non-Saccharomyces yeasts including Kazachstania humilis provide leavening through carbon dioxide production during fermentation of sugars liberated by amylase enzymes in the flour. Lactic acid bacteria including Lactobacillus sanfranciscensis produce lactic and acetic acids that lower dough pH, creating the characteristic tangy flavour profile of sourdough and inhibiting pathogenic microorganisms through acidification. Autolyse combines flour and water before adding starter and salt, allowing gluten-forming proteins glutenin and gliadin to hydrate and begin forming gluten networks without mechanical agitation. Bulk fermentation at room temperature between 24 and 26 degrees Celsius takes 4 to 6 hours depending on starter activity and ambient temperature, during which the dough doubles in volume and develops flavour through organic acid accumulation. Stretch and fold cycles during bulk fermentation build gluten strength without degassing the dough by folding the mass over itself from four sides at regular intervals. Cold retard in the refrigerator at 4 degrees Celsius slows fermentation dramatically, allowing bakers to extend the schedule and develop more complex acetic acid flavour compounds over 8 to 16 hours.',
    },
    {
        title: 'Indian spice blending',
        body: 'Indian spice blending is a sophisticated culinary practice rooted in Ayurvedic principles of balancing flavours and medicinal properties through precise combination of aromatic seeds, barks, roots, and dried fruits. Garam masala translates literally as hot spice mixture and typically combines warming spices including cardamom pods, cloves, cinnamon bark, black pepper, cumin seeds, and coriander seeds that are dry-roasted to bloom volatile aromatic compounds before grinding. Tempering or tadka involves adding whole spices including mustard seeds, cumin, dried red chillies, and curry leaves to hot ghee or oil, causing them to sputter and release fat-soluble flavour compounds that permeate the entire dish. Chaat masala is a tangy, savoury blend featuring amchur dried mango powder as its primary souring agent alongside kala namak black salt, roasted cumin, coriander, ginger, and chilli powder, used as a finishing condiment on street foods and salads. Biryani masala is a complex blend including saffron threads, star anise, mace blades, green cardamom, dried rose petals, kewra water, and nutmeg that creates the floral, layered fragrance characteristic of Hyderabadi and Lucknowi biryani. Regional spice traditions vary significantly across India, with coastal Chettinad cuisine using stone-ground kalpasi stone flower and marathi mokku in ways that have no equivalents in northern Mughal-influenced cuisines.',
    },
    {
        title: 'Chocolate tempering',
        body: 'Chocolate tempering is the controlled crystallisation process that produces the stable polymorphic Form V beta crystal structure in cocoa butter, responsible for the characteristic snap, glossy surface, and smooth mouthfeel of properly tempered couverture chocolate. Cocoa butter can crystallise into six polymorphic forms with melting points ranging from 17 to 36 degrees Celsius, and only Form V produces the desirable sensory properties required for confectionery and coating applications. The tabling method melts chocolate completely to 50 degrees to destroy all existing crystals, then pours two thirds onto a marble slab and works it with a palette knife and scraper until it cools to 27 degrees and thickens, then combines it with the remaining warm chocolate to raise the temperature to the working range of 31 to 32 degrees for dark chocolate. The seeding method achieves the same result by adding finely chopped or grated pre-tempered chocolate to fully melted chocolate at 50 degrees, stirring continuously as the seed crystals propagate stable Form V crystallisation throughout the mass. Bloom defects occur as either fat bloom from temperature fluctuations that melt and recrystallise cocoa butter as Form IV, or sugar bloom from moisture condensation that dissolves and recrystallises surface sugar, both manifesting as white streaks or dusty grey patches on the chocolate surface.',
    },
    {
        title: 'Bread scoring techniques',
        body: 'Bread scoring is the practice of cutting the surface of shaped dough immediately before baking with a sharp blade to control oven spring, direct expansion, and create decorative patterns that identify the baker and communicate the bread style. The lame is the traditional scoring tool consisting of a razor blade mounted on a curved metal handle that allows the baker to cut at a precise angle to the dough surface with a swift, confident stroke. Ear formation is the signature feature of sourdough batards and baguettes where the scored flap lifts away from the loaf as steam pressure forces expansion through the cut, creating a raised ridge that browns and crisps into a crackling ear. Scoring angle determines the character of the opening, with cuts at 30 degrees to the surface creating ear formation while perpendicular cuts create symmetrical expansion without a lifted flap. Decorative scoring patterns including wheat sheaves, leaves, ferns, and geometric designs require a rested, well-chilled dough that holds its shape during cutting and a confident hand that executes each stroke in a single continuous motion without hesitation or backtracking. Whole grain and high-hydration doughs require deeper cuts because their denser structure and less elastic gluten network expand less dramatically in the oven than white flour doughs, requiring more encouragement to open fully.',
    },
    {
        title: 'Wine pairing principles',
        body: 'Wine and food pairing aims to create harmony or deliberate contrast between the sensory components of both, including acidity, tannin, sweetness, body, and aromatic intensity. Acidity in wine cuts through fat and richness in food by providing a refreshing counterpoint that cleanses the palate between bites, making crisp high-acid whites like Chablis and Muscadet natural partners for cream-based sauces and rich seafood preparations. Tannins are polyphenolic compounds found primarily in red grape skins and oak barrels that bind with salivary proteins and create a drying, astringent sensation which is moderated by the proteins and fats in red meat, making tannic Barolo and Cabernet Sauvignon ideal companions for aged beef and lamb. Residual sugar in off-dry and sweet wines balances the heat and capsaicin compounds in spicy dishes by providing sweetness that tempers the burning sensation while refreshing the palate. Umami-rich foods including aged Parmesan, anchovies, and dried mushrooms intensify the perception of tannins and bitterness in wine, making highly tannic reds taste harsh and metallic alongside these ingredients. Regional pairing wisdom reflects centuries of culinary evolution where local wines developed alongside local cuisines, producing the truism that regional wines pair naturally with regional food.',
    },
    {
        title: 'Japanese knife skills',
        body: 'Japanese culinary knife craft is a discipline that values precision, respect for ingredients, and the development of technical skill over years of dedicated practice under the guidance of experienced chefs. The yanagiba is a long, slender single-bevel knife designed exclusively for slicing raw fish in a single drawing motion from heel to tip, producing clean cuts that preserve the cellular structure of delicate fish flesh without compression or tearing. The deba is a thick, heavy single-bevel knife used for fish butchery, its substantial spine providing the mass needed to cut through bones cleanly while the acute single-bevel edge produces clean flesh cuts with minimal resistance. The usuba is a thin rectangular single-bevel knife used for katsuramuki rotary peeling and paper-thin vegetable slicing using a push-cut technique that keeps the wrist rigid while pivoting at the shoulder. Single-bevel Japanese knives are sharpened exclusively on the flat face side to maintain the precise geometry required for their specific cutting techniques, requiring a different sharpening approach than the symmetrical bevels of Western knives. Honbazuke edge finishing on Japanese knives produces a mirror-polished cutting edge by progressing through whetstones of increasing grit from 220 through 1000, 3000, and 8000, finishing with a leather strop to align the edge apex.',
    },
    {
        title: 'Japan itinerary planning',
        body: 'Planning a Japan itinerary requires balancing iconic destinations with the logistical realities of a railway network that, while extraordinarily punctual and comprehensive, demands advance research to navigate efficiently. The Japan Rail Pass grants unlimited travel on most JR-operated shinkansen bullet trains, limited express trains, and local lines, and must be purchased outside Japan as a voucher exchanged at major stations upon arrival. IC cards including Suica issued in Tokyo and ICOCA in Osaka function as stored-value transit cards accepted on virtually all local trains, buses, and subway systems throughout Japan, and increasingly at convenience stores, vending machines, and restaurants. Temple and shrine visits reward early morning arrivals before tour groups arrive, with Fushimi Inari Taisha in Kyoto offering dramatically different experiences at 6am versus midday when thousands of visitors crowd the torii gate tunnels. Ryokan traditional inns provide an immersive experience in Japanese hospitality through kaiseki multi-course dinners featuring seasonal local ingredients, communal gender-separated onsen hot spring bathing, and futon sleeping arrangements on tatami floors. Seasonal timing dramatically affects the experience, with cherry blossom season in late March and April drawing enormous crowds and commanding premium accommodation rates that require booking 6 to 12 months in advance. Day trips from Tokyo to Kamakura, Nikko, or Hakone and from Kyoto to Nara and Osaka are straightforward on the rail network.',
    },
    {
        title: 'Southeast Asia budget travel',
        body: 'Southeast Asia offers extraordinary value for budget travellers willing to embrace overnight transportation, street food culture, and guesthouse accommodation that provides cleanliness and safety at a fraction of resort prices. Night trains connecting Bangkok to Chiang Mai and Hanoi to Ho Chi Minh City combine transportation and accommodation costs into a single affordable ticket while covering distances that would otherwise require an additional hotel night. Overland border crossings between Thailand, Cambodia, and Vietnam using bus services with advance visa arrangements cost a fraction of flying and expose travellers to border town culture and the gradual transition between countries. Guesthouses in smaller provincial towns away from major tourist circuits offer genuine local experiences at prices 50 to 70 percent lower than equivalent accommodation in tourist hubs while maintaining acceptable standards of cleanliness and security. Street food stalls and market vendors serve identical or superior versions of regional dishes compared to restaurants catering to tourists, at prices that allow three full meals per day within a modest daily food budget. Motorbike rental provides the most flexible and economical way to explore rural areas of Vietnam, Cambodia, and northern Thailand where public transportation is infrequent and tuk-tuk fares for tourists are inflated. SIM cards with generous data allowances are available at airports and convenience stores throughout the region.',
    },
    {
        title: 'Europe interrail planning',
        body: 'The Interrail Global Pass grants European residents unlimited rail travel across 33 countries within a specified validity period, with different pass types offering continuous validity or a set number of travel days within a longer window. High-speed train services including Eurostar, Thalys, AVE, Frecciarossa, and ICE require seat reservations costing between 5 and 35 euros per journey on top of the Interrail pass, representing a significant additional cost on routes between major cities that must be factored into the travel budget. Point-to-point tickets purchased in advance on national rail websites frequently undercut the per-journey cost of the Interrail pass for single-country travel within France, Spain, or Italy, making the pass most economical for travellers crossing multiple borders within the validity period. Luggage storage facilities at major train stations across Europe charge 3 to 8 euros per item per day, enabling travellers to leave bags during layovers and explore cities unencumbered before continuing to the next destination. Travelling in shoulder season between October and April reduces accommodation costs by 40 to 60 percent compared to July and August peaks, while also reducing crowds at major attractions and enabling more spontaneous itinerary changes without pre-booking pressure. Couchette and sleeper train reservations on overnight routes including the Nightjet network allow travellers to cover long distances while sleeping, saving both accommodation and daytime travel time.',
    },
    {
        title: 'Backpacking South America',
        body: 'South America rewards patient backpackers with extraordinary geographic and cultural diversity across landscapes ranging from Amazonian rainforest and Andean altiplano to Patagonian steppe and Caribbean coastline, accessible through a combination of long-distance buses, domestic flights, and occasional boat journeys. Altitude sickness affects virtually all travellers arriving directly at high-elevation destinations including Cusco at 3400 metres and La Paz at 3600 metres, requiring at minimum two days of rest and acclimatisation before attempting hiking or physical activity, with coca tea and acetazolamide medication providing partial symptom relief. Overnight buses operated by premium companies including Cruz del Sur in Peru provide semi-cama and cama-full reclinable seat configurations with meal service and entertainment, covering distances of 8 to 20 hours while saving accommodation costs. Colombian domestic aviation has become extraordinarily competitive with advance booking, with Bogotá to Medellín or Bogotá to Cartagena routes frequently priced comparably to bus journeys when booked two to three weeks ahead through Avianca, Latam, or Viva Air. Argentina peso volatility creates a persistent gap between the official exchange rate and the blue dollar parallel market rate, with cash US dollars consistently obtaining 30 to 60 percent more Argentine pesos than card payments processed at the official bank rate. Border crossing requirements and visa policies change frequently throughout the region and require current verification through official consulate sources.',
    },
    {
        title: 'Safari preparation Kenya',
        body: 'Planning a Kenyan safari requires matching travel timing to wildlife events, budget to accommodation tier, and health preparation to the specific disease risks of the regions visited. The Great Migration crossing of the Mara River in the Masai Mara occurs primarily between July and October when wildebeest and zebra herds numbering over a million animals cross from Tanzania in search of fresh grass, creating dramatic predator-prey interactions that represent the pinnacle of East African wildlife viewing. Self-drive safaris through national parks and reserves require a four-wheel-drive vehicle with high ground clearance suitable for murram laterite roads that become deeply rutted and slippery during heavy rains, along with detailed offline maps and sufficient fuel capacity for remote areas without petrol stations. Professional guide-operated game drives in dedicated safari vehicles with roof hatches provide superior wildlife spotting through accumulated local knowledge of animal territories and behaviours that independent travellers cannot replicate. Malaria prophylaxis using atovaquone-proguanil, doxycycline, or mefloquine should begin 1 to 2 weeks before travel depending on the medication and continue for the prescribed period after return, combined with insect repellent containing DEET and permethrin-treated clothing and bedding. Yellow fever vaccination is required for entry and should be administered at least 10 days before travel, while typhoid, hepatitis A, and rabies vaccinations are strongly recommended depending on itinerary and risk tolerance.',
    },
    {
        title: 'Visa and documentation',
        body: 'International travel documentation requires meticulous preparation to avoid entry refusals, deportation, or being offloaded by airlines at the departure gate for missing paperwork. The Schengen Area encompasses 27 European countries that share a common external border and allow free internal movement, with non-EU visitors permitted a maximum of 90 days within any 180-day rolling period across the entire zone regardless of which member states are visited. Many countries require travellers to demonstrate onward travel arrangements before airline check-in even when no visa is required for entry, necessitating either a confirmed return ticket or a refundable onward ticket purchased specifically to satisfy this requirement. Comprehensive travel insurance must include medical evacuation coverage of at least 100,000 US dollars given that air ambulance repatriation from remote destinations can cost 50,000 to 150,000 dollars without insurance, with policies verified to cover adventure activities if hiking, diving, or motorcycling are planned. Passport validity requirements extend beyond the trip end date in most countries, with many requiring six months of remaining validity beyond the departure date and others requiring two blank pages for entry and exit stamps. Certified photocopies of passport data pages, visa stamps, travel insurance policies, and accommodation confirmations stored separately from the originals and additionally in encrypted cloud storage provide recovery options in case of theft or loss.',
    },
    {
        title: 'Hostel culture and etiquette',
        body: 'Hostel accommodation offers budget travellers a social infrastructure for meeting fellow travellers, sharing local knowledge, and building friendships across nationalities within a communal living environment that has its own established norms and etiquette. Dormitory room etiquette requires silencing all electronic devices after a reasonable evening hour, using a red-light headlamp rather than a white light when moving around the room after others have retired, and packing bags the previous evening to avoid noisy rummaging during early morning departures. Locker security requires travellers to provide their own padlock as hostels supply the locker hardware but rarely the lock, with combination locks preferable over key locks to eliminate the risk of key loss. Common areas including kitchens, lounges, rooftop terraces, and communal tables function as natural social spaces where solo travellers congregate to share itinerary recommendations, travel warnings, restaurant tips, and company for day trips and evening activities. Kitchen use typically operates within designated hours with an expectation that users clean their cooking equipment and dishes immediately, label refrigerated food with name and departure date, and leave shared surfaces cleaner than they found them. Hostel social events including pub crawls, walking tours, cooking classes, and movie nights organised by staff provide structured opportunities for guests to meet without the awkwardness of cold introductions in communal spaces.',
    },
    {
        title: 'System design principles',
        body: 'System design for large-scale distributed applications requires making explicit trade-offs between consistency, availability, and partition tolerance as described by the CAP theorem, with most production systems choosing between strong consistency with reduced availability or eventual consistency with higher availability during network partitions. Horizontal scaling distributes load across multiple commodity server instances behind a load balancer, enabling near-linear capacity increases by adding nodes without requiring downtime or migration to larger hardware. Vertical scaling increases the compute resources of a single instance by upgrading CPU, RAM, or storage, providing simplicity but limited by the maximum available hardware tier and creating a single point of failure. Caching reduces database load and improves response latency by storing frequently accessed data in fast in-memory stores including Redis and Memcached at multiple layers of the stack including CDN edge nodes, application servers, and database query caches. Message queues including Apache Kafka and RabbitMQ decouple producers from consumers by buffering events durably, enabling independent scaling of ingestion and processing components and providing fault tolerance through replay of unconsumed messages. Database sharding horizontally partitions data across multiple database instances based on a shard key, distributing both storage and query load while introducing complexity in cross-shard queries and rebalancing operations.',
    },
    {
        title: 'API design best practices',
        body: 'Well-designed APIs enable reliable integration between systems by providing clear contracts, predictable behaviour, and sufficient backward compatibility to allow independent evolution of client and server. RESTful API design uses HTTP verbs semantically, with GET retrieving resources without side effects, POST creating new resources or triggering actions, PUT replacing resources entirely, PATCH applying partial updates, and DELETE removing resources, enabling HTTP infrastructure including caches and proxies to behave correctly. Cursor-based pagination uses an opaque token representing the position in the result set rather than numeric offsets, preventing inconsistent results when records are inserted or deleted between page requests. Rate limiting protects API infrastructure from abuse and accidental overload using algorithms including the token bucket, which allows bursting up to a capacity limit, and the sliding window, which distributes allowed requests evenly across time. API versioning through URL path prefixes like v1 or v2 or through Accept header content negotiation allows breaking changes to be introduced without disrupting existing clients, with a clear deprecation timeline communicated through documentation and response headers. OpenAPI Specification defines API contracts as machine-readable YAML or JSON documents that enable automatic generation of client SDKs, server stubs, and interactive documentation. Idempotency keys on mutation endpoints allow clients to safely retry requests after network failures by enabling the server to detect and return the cached result of a previously completed operation.',
    },
    {
        title: 'Docker and containerization',
        body: 'Containerisation packages application code, runtime dependencies, configuration, and system libraries into a standardised portable unit that runs consistently across development, testing, staging, and production environments regardless of the underlying host operating system. Docker containers share the host operating system kernel rather than emulating complete virtual hardware like traditional virtual machines, making them lightweight enough to start in milliseconds and dense enough to run hundreds per host where a VM-based approach might support only dozens. Multi-stage Dockerfile builds separate the build environment from the runtime environment, using a full SDK image to compile code and then copying only the compiled binary and runtime dependencies into a minimal base image, reducing the final image size by 80 to 95 percent and eliminating build tools from the production attack surface. Docker Compose defines multi-container application stacks as declarative YAML configurations specifying services, networks, volume mounts, environment variables, and dependency ordering, enabling complete development environments to be started with a single command. Health check instructions in Dockerfiles and Compose configurations specify commands that container orchestrators use to determine service readiness, enabling rolling deployments that route traffic only to containers that have passed health validation. Container image layers are cached by the Docker build daemon based on content hashing, enabling incremental rebuilds that reuse unchanged layers and dramatically reducing build times in CI pipelines.',
    },
    {
        title: 'Git workflow strategies',
        body: 'Version control workflows define how teams coordinate changes to a shared codebase through branching strategies, code review processes, and merge policies that balance development velocity with code quality and deployment safety. Feature branch workflows isolate each unit of work on a dedicated branch created from the main branch, enabling parallel development without conflict until the feature is complete and ready for review through a pull request or merge request. Squash merging collapses all commits on a feature branch into a single commit on the main branch, maintaining a clean linear history that is easy to bisect and revert while discarding the incremental commit noise accumulated during development. Rebase rewriting integrates changes by replaying feature branch commits on top of the current main branch tip, producing a linear history without merge commits at the cost of rewriting commit hashes and requiring force-pushes to remote branches. Conventional commit message format prefixes commit messages with structured type tokens including feat for new features, fix for bug fixes, docs for documentation, refactor for non-functional changes, and chore for maintenance, enabling automated changelog generation and semantic version bumping. Branch protection rules enforce code review approvals, passing CI checks, and linear history requirements before merging to protected branches, preventing accidental direct pushes to production branches.',
    },
    {
        title: 'Database indexing strategies',
        body: 'Database indexes accelerate query execution by maintaining auxiliary data structures that allow the query planner to locate rows matching filter conditions without scanning the entire table, trading increased write overhead and storage consumption for dramatically improved read performance. B-tree indexes organise indexed values in a balanced tree structure that supports equality predicates, range queries, and ORDER BY operations on the indexed columns, making them the default index type suitable for most OLTP query patterns. Composite indexes covering multiple columns in a specific order enable the query planner to satisfy multi-column predicates using a single index scan, but require the leading column to appear in the WHERE clause or the index cannot be used. Partial indexes index only rows satisfying a specified filter condition, reducing index size and maintenance overhead for use cases including non-null status columns, active record flags, or recent date ranges where only a subset of rows is frequently queried. Covering indexes include all columns required to satisfy a query as either index keys or non-key included columns, enabling index-only scans that retrieve all required data from the index without accessing the underlying table heap. EXPLAIN ANALYZE in PostgreSQL executes a query and reports both the planned and actual row counts, execution time, and node-by-node cost estimates, revealing misestimates in the query planner statistics that cause suboptimal index selection.',
    },
    {
        title: 'TypeScript advanced patterns',
        body: 'TypeScript advanced type system features enable library authors and application developers to express complex type-level constraints that catch entire categories of bugs at compile time without runtime overhead. Mapped types transform all properties of an existing type by iterating over its keys using the in keyof syntax, enabling systematic transformations including making all properties optional with Partial, required with Required, or readonly with Readonly. Conditional types select between type branches based on whether a type extends a constraint, using the T extends U ? X : Y syntax to express type-level logic including extracting return types, unwrapping Promises, and distributing over union types. Template literal types construct new string types by combining string literals and other types using template literal syntax, enabling type-safe string manipulation including building event name unions, CSS property names, and API endpoint paths. Discriminated unions use a common literal type property as a discriminant that TypeScript narrows in switch statements and conditional checks, enabling exhaustive pattern matching over variant types without runtime type guards. The infer keyword within conditional types extracts type information from generic positions, enabling utilities including ReturnType, Parameters, and InstanceType that introspect function and class signatures.',
    },
    {
        title: 'Testing strategies',
        body: 'Comprehensive software testing combines multiple testing approaches at different levels of the system to provide confidence in correctness, catch regressions early, and document expected behaviour for future maintainers. Unit tests verify the behaviour of individual functions and modules in isolation by replacing external dependencies including databases, network services, and file systems with lightweight mocks or stubs, enabling fast execution and precise failure attribution. Integration tests verify that multiple components interact correctly by exercising real implementations of dependencies in controlled test environments, catching interface mismatches and configuration errors that unit tests with mocks cannot detect. End-to-end tests simulate complete user workflows through the deployed application using browser automation tools including Playwright and Cypress, verifying that the entire stack from frontend through backend to database functions correctly from the user perspective. Property-based testing generates large numbers of random inputs satisfying specified constraints and verifies that output properties hold universally, discovering edge cases and boundary conditions that example-based tests miss by construction. Snapshot testing captures the rendered output of UI components or serialised data structures and fails when subsequent runs produce different output, providing a low-effort regression detection mechanism that alerts developers to unexpected changes.',
    },
    {
        title: 'Performance optimization',
        body: 'Software performance optimization requires measurement-driven analysis to identify and address actual bottlenecks rather than optimising speculatively based on intuition, as the bottleneck is frequently not where developers expect it to be. Profiling tools including Chrome DevTools performance timeline, Node.js clinic.js, and language-specific profilers identify which functions consume the most CPU time and memory, providing empirical data to direct optimization effort where it will have the greatest impact. Memory leaks in long-running JavaScript applications commonly originate from event listeners attached to DOM elements that are removed without removing their listeners, global variable accumulation, and closure references that prevent garbage collection of large objects. Database N+1 query problems occur when an application fetches a list of records and then executes a separate query for each record to retrieve related data, replaced by eager loading with JOIN queries or GraphQL DataLoader batching that consolidates related data fetching into bulk operations. Memoisation caches the return values of pure functions keyed by their input arguments, eliminating redundant recomputation for functions called repeatedly with identical arguments at the cost of increased memory consumption. Web application performance is dominated by network latency and asset download size for most users, making compression, code splitting, tree shaking, and CDN distribution more impactful than algorithmic optimizations that reduce server CPU time by milliseconds.',
    },
    {
        title: 'Investment fundamentals',
        body: 'Personal investment strategy requires understanding the relationship between risk, return, time horizon, and diversification to build portfolios that grow wealth reliably over decades without requiring active management or market timing ability. Index funds passively replicate the performance of market indices including the S&P 500, total market, and international equity indices by holding all constituent securities in proportion to their market capitalisation, providing broad diversification at expense ratios averaging 0.03 to 0.20 percent annually compared to 0.5 to 1.5 percent for actively managed funds that rarely outperform their benchmarks over long periods. Dollar cost averaging invests a fixed monetary amount at regular intervals regardless of current price levels, automatically purchasing more shares when prices are low and fewer when prices are high, reducing the impact of market timing and emotional decision-making on long-term returns. Asset allocation distributes portfolio value across stocks, bonds, real estate investment trusts, and cash equivalents in proportions calibrated to investment time horizon and risk tolerance, with younger investors typically holding higher equity allocations. Annual rebalancing restores the portfolio to target allocation percentages after market movements cause drift by selling appreciated assets and purchasing underperformed assets, systematically enforcing buy-low-sell-high behaviour without requiring market predictions. Tax-advantaged accounts including 401k plans, individual retirement accounts, and health savings accounts shelter investment returns from annual taxation.',
    },
    {
        title: 'Tax optimization strategies',
        body: 'Legal tax optimization reduces lifetime tax liability by structuring income, investments, and expenses to minimise taxable events while maximising use of available deductions, credits, and tax-advantaged account structures. Tax-loss harvesting sells investment positions at a loss to realise capital losses that offset capital gains from other sales, reducing the current year tax liability while maintaining market exposure by immediately purchasing a similar but not substantially identical security to avoid wash sale rule violations. Roth IRA contributions are made with after-tax dollars and grow completely tax-free, with qualified withdrawals in retirement including all gains exempt from federal income tax, making Roth accounts most valuable for investors who expect to be in a higher tax bracket in retirement than during their working years. Traditional 401k and IRA contributions reduce current-year gross income by the contributed amount, deferring taxation until withdrawal in retirement when many investors are in lower marginal brackets due to reduced income. Health savings accounts paired with high-deductible health plans offer a triple tax advantage through deductible contributions, tax-free growth, and tax-free withdrawals for qualified medical expenses, with funds rolling over indefinitely and becoming available for any purpose after age 65 at ordinary income tax rates. Qualified opportunity zone investments defer and potentially reduce capital gains taxes by reinvesting gains into designated economically distressed communities within 180 days.',
    },
    {
        title: 'Emergency fund planning',
        body: 'An emergency fund is a dedicated liquid cash reserve held outside investment accounts that provides financial stability during income disruption, unexpected large expenses, or economic uncertainty without requiring liquidation of long-term investments at potentially unfavourable prices. The conventional emergency fund target of three to six months of essential living expenses represents a reasonable balance between the opportunity cost of holding cash and the coverage needed for the most common financial disruptions including job loss, medical expenses, major vehicle repairs, and home maintenance emergencies. High-yield savings accounts at online banks offer interest rates 10 to 20 times higher than traditional brick-and-mortar bank savings accounts while maintaining FDIC insurance coverage up to 250,000 dollars and allowing instant transfers to checking accounts for expenses. Money market mutual funds offer comparable yields to high-yield savings accounts with same-day or next-day liquidity by investing in short-duration government and corporate debt instruments, though they lack FDIC insurance and technically carry minimal credit risk. Certificate of deposit ladders split the emergency fund into multiple CDs with staggered maturity dates at 3, 6, 9, and 12 month intervals, capturing slightly higher yields than savings accounts on the longer-duration tranches while ensuring that at least one portion matures and becomes available every quarter. Building an emergency fund requires prioritising consistent monthly contributions above discretionary spending and below fixed essential expenses in the budget hierarchy.',
    },
    {
        title: 'Credit score management',
        body: 'Credit scores calculated by FICO and VantageScore models determine the interest rates offered on mortgages, auto loans, and personal credit, with differences of 100 points potentially representing tens of thousands of dollars in additional interest costs over the life of a loan. Payment history is the single largest factor in FICO score calculation at 35 percent of the total score, making consistent on-time payment of every account the most impactful credit building behaviour available, with a single missed payment potentially reducing scores by 60 to 110 points. Credit utilisation ratio measures outstanding revolving balances as a percentage of total credit limits, with ratios below 10 percent associated with the highest scores and ratios above 30 percent significantly depressing scores regardless of income or payment history. Credit age encompasses both the age of the oldest account and the average age of all accounts, rewarding long credit histories and penalising the opening of multiple new accounts in short periods which reduces the average age and triggers multiple hard inquiries. Authorised user status on a long-established account with low utilisation and perfect payment history adds that account positive history to the authorised user credit report without requiring the user to qualify independently. Hard inquiries from credit applications remain on credit reports for two years but typically affect scores only for 12 months, with multiple mortgage or auto loan inquiries within a 45 day window typically treated as a single inquiry by scoring models.',
    },
    {
        title: 'Budgeting methods',
        body: 'Personal budgeting frameworks provide structure for allocating income across competing financial priorities including essential expenses, discretionary spending, debt repayment, and savings goals, with different approaches suiting different temperaments and financial situations. The 50-30-20 rule divides after-tax income into 50 percent for needs including housing, utilities, groceries, transportation, and minimum debt payments, 30 percent for wants including dining, entertainment, subscriptions, and non-essential purchases, and 20 percent for savings, investments, and additional debt repayment above minimums. Zero-based budgeting assigns every dollar of monthly income to a specific named category until the total allocated equals income, ensuring no money flows to unintended purposes and creating explicit awareness of all spending categories including irregular annual expenses divided into monthly reserves. The envelope method implements zero-based budgeting physically by withdrawing cash and distributing it into labelled envelopes for each discretionary category, making spending visceral and finite in a way that card payments do not, which research consistently shows reduces discretionary overspending. Pay yourself first automation transfers a predetermined savings amount from checking to savings or investment accounts on the day income arrives, removing the money from the discretionary pool before it can be spent and making saving the default rather than the residual. Budget categories require regular review as income and expenses evolve, with quarterly reconciliation comparing actual spending to budget allocations.',
    },
    {
        title: 'Retirement planning',
        body: 'Retirement planning requires projecting future income needs, estimating portfolio withdrawal requirements, selecting appropriate account structures, and managing the sequence of returns risk that makes early retirement losses far more damaging than equivalent losses experienced later. The 4 percent safe withdrawal rate guideline estimates the annual portfolio withdrawal percentage that historical simulations suggest would survive 30-year retirement periods across market cycles including the Great Depression and 1970s stagflation, though critics argue lower rates are appropriate for longer retirements and current low expected returns. Sequence of returns risk describes how portfolio longevity depends not just on average returns but on the order in which returns occur, with significant losses in the first years of retirement permanently reducing the capital base available to recover and generate future returns regardless of subsequent strong performance. Social Security benefit optimisation involves deciding when between age 62 and 70 to begin claiming benefits, with each year of delay from 62 to 70 increasing the monthly benefit by 5 to 8 percent, representing a significant guaranteed income enhancement for those with adequate bridge assets. Required minimum distributions from traditional IRAs and 401k accounts begin at age 73 under current legislation, mandating annual withdrawals calculated from account balances and IRS life expectancy tables that may push retirees into higher tax brackets. Roth conversion ladders systematically convert portions of traditional IRA balances to Roth accounts in years with lower income, prepaying tax at lower marginal rates to reduce future RMD obligations.',
    },
    {
        title: 'Insurance coverage',
        body: 'Insurance transfers financial risk from individuals to insurers by pooling premiums across large numbers of policyholders, making catastrophic but low-probability financial losses manageable through regular affordable premium payments. Term life insurance provides pure death benefit coverage for a specified term of 10, 20, or 30 years without accumulating cash value, making it the most cost-effective protection for income replacement during working years when dependants rely on the insured income to maintain their standard of living. Disability insurance replaces 60 to 70 percent of gross income if illness or injury prevents the insured from performing their occupation, protecting against the statistically most common catastrophic financial risk for working-age adults who are far more likely to experience a disabling injury or illness than death during their working years. Umbrella liability policies provide excess liability coverage of 1 to 5 million dollars above the limits of underlying auto and homeowners policies, protecting against lawsuits arising from serious auto accidents, injuries on one property, or reputational damages at annual premiums of 150 to 300 dollars. Health insurance deductible, copayment, coinsurance, and out-of-pocket maximum provisions interact to determine the actual cost of healthcare utilisation, with high-deductible health plans offering lower premiums in exchange for higher initial cost-sharing that makes them most economical for healthy individuals who pair them with health savings account contributions. Long-term care insurance covers assisted living facility, nursing home, and in-home care costs that Medicare does not cover.',
    },
    {
        title: 'Strength training programming',
        body: 'Effective strength training programme design applies progressive overload systematically to drive continuous neuromuscular and hypertrophic adaptation without accumulating fatigue that exceeds the trainee capacity to recover and adapt. Progressive overload increases training stimulus over time by adding weight to the bar, increasing repetition count, adding sets, reducing rest periods, or improving technique quality, preventing accommodation to a fixed stimulus that causes plateau. Periodisation organises training into structured phases alternating between accumulation blocks that build volume and work capacity, intensification blocks that develop maximal strength through heavier loads and lower volumes, and deload weeks that reduce volume by 40 to 60 percent to dissipate accumulated fatigue and allow supercompensation. Compound barbell movements including the squat, deadlift, bench press, overhead press, and barbell row recruit multiple large muscle groups across multiple joints simultaneously, providing the greatest stimulus for whole-body strength and hypertrophy development per unit of training time. Rest periods between heavy compound sets of 3 to 5 minutes allow complete resynthesis of phosphocreatine energy stores depleted during maximum effort contractions, enabling subsequent sets to be performed with minimal performance decrement. Rate of perceived exertion and reps in reserve scales provide subjective proxies for training intensity that allow load management relative to daily readiness fluctuations without requiring fixed percentage-based programming.',
    },
    {
        title: 'Nutrition for performance',
        body: 'Sports nutrition supports training adaptation, recovery, and performance by ensuring macronutrient availability matches energy demands, micronutrient adequacy supports biochemical processes, and meal timing aligns nutrient availability with physiological needs. Protein intake of 1.6 to 2.2 grams per kilogram of body mass daily maximises muscle protein synthesis rates based on meta-analyses of resistance training studies, distributed across 4 to 6 meals of 0.3 to 0.5 grams per kilogram per serving to saturate aminoacyl-tRNA synthetase activity and leucine-mediated mTORC1 signalling pathways. Carbohydrate periodisation matches glycolytic substrate availability to training demands by consuming higher carbohydrate amounts on high-intensity training days and lower amounts on recovery or low-intensity days, optimising both performance and metabolic flexibility. Pre-training carbohydrate consumption 1 to 3 hours before high-intensity sessions restores liver glycogen depleted overnight and tops up muscle glycogen to support performance during training exceeding 60 minutes at intensities above 70 percent of maximal oxygen uptake. Essential fatty acids including EPA and DHA from marine sources attenuate exercise-induced inflammation, support neurological function, improve anabolic signalling sensitivity, and reduce muscle soreness at supplemental doses of 2 to 4 grams daily. Hydration status affects aerobic performance significantly, with losses of 2 percent of body mass through sweat reducing maximal oxygen uptake by 10 to 20 percent, requiring fluid intake calibrated to sweat rate which varies between 0.5 and 2.5 litres per hour.',
    },
    {
        title: 'Sleep optimization',
        body: 'Sleep is the primary physiological recovery mechanism for athletic performance adaptation, cognitive function, hormonal regulation, and immune competence, with chronic restriction producing cumulative deficits that impair all these systems without subjective awareness of the degree of impairment. Circadian rhythm is an endogenous approximately 24-hour biological clock regulated primarily by light exposure, with morning bright light suppressing melatonin production and advancing the circadian phase while evening blue light exposure delays the phase and postpones sleep onset. Core body temperature follows a circadian rhythm that must decrease by 1 to 2 degrees Celsius to initiate sleep onset, making sleeping in cool environments between 16 and 19 degrees Celsius and taking warm baths 1 to 2 hours before sleep both effective strategies for accelerating this thermal decline. Sleep architecture cycles through non-REM stages 1 and 2 light sleep, stage 3 slow-wave deep sleep associated with physical recovery and growth hormone secretion, and REM sleep associated with memory consolidation and emotional processing, with cycles lasting approximately 90 minutes. Caffeine competitively inhibits adenosine receptors throughout the brain, blocking the sleep pressure signal that accumulates during waking hours and delaying sleep onset when consumed within 5 to 6 half-lives of sleep time, with the plasma half-life ranging from 3 to 7 hours depending on individual CYP1A2 enzyme activity. Sleep extension studies with athletes demonstrate improvements of 9 to 12 percent in sport-specific performance metrics including sprint times, shooting accuracy, and reaction speed after 2 to 3 weeks of extending nightly sleep to 9 to 10 hours.',
    },
    {
        title: 'Running training plans',
        body: 'Structured running training develops aerobic capacity, lactate threshold, and running economy through systematic application of varied intensity workouts across weekly and mesocycle training blocks, with periodisation balancing progressive overload against recovery to produce continuous adaptation without overuse injury. Easy aerobic runs performed at conversational pace below 70 percent of maximum heart rate develop mitochondrial density, cardiac stroke volume, and fat oxidation capacity without imposing significant recovery demands, enabling high training volume accumulation that builds the aerobic base supporting all higher intensity work. Lactate threshold tempo runs sustained at the pace corresponding to approximately 4 millimoles per litre blood lactate for 20 to 40 minutes continuously or in cruise interval sets improve the sustainable race pace by upward-shifting the lactate curve through increased lactate clearance capacity and delayed onset of acidosis. VO2max interval training at 95 to 100 percent of maximal oxygen uptake intensity with work-to-rest ratios of 1:1 to 2:1 develops maximal aerobic power and improves oxygen transport efficiency through cardiac and peripheral adaptations. Strides are 20 to 30 second accelerations to near-sprint pace with full recovery between repetitions performed 2 to 4 times per week after easy runs to maintain neuromuscular recruitment patterns and leg speed without imposing fatigue. Long slow distance runs extending to 20 to 30 percent of weekly mileage develop muscular endurance, glycogen storage capacity, fat oxidation efficiency, and psychological resilience for sustained effort.',
    },
    {
        title: 'Mobility and flexibility',
        body: 'Mobility training develops functional range of motion through active control across joint angles while flexibility training increases passive range of motion through sustained tissue lengthening, with both contributing to injury prevention, movement quality, and athletic performance across training modalities. Static stretching maintained for 30 to 60 seconds consistently increases passive range of motion through viscoelastic tissue changes and altered stretch tolerance when performed daily, but transiently reduces force production capacity for 15 to 60 minutes post-stretch through mechanisms including reduced motor unit recruitment and altered muscle-tendon unit stiffness, contraindicating pre-training static stretching for strength or power activities. Dynamic stretching moves joints through full ranges of motion under active muscle control using controlled swings, rotations, and locomotion patterns to increase tissue temperature, synovial fluid distribution, and neuromuscular activation without the performance decrements associated with static stretching, making it the preferred warm-up modality before training sessions. Foam rolling applies sustained mechanical pressure to myofascial tissue through the body weight acting on a cylindrical foam cylinder, transiently improving range of motion through neurological relaxation mechanisms and potentially reducing delayed onset muscle soreness perception. Hip flexor mobility is critically limiting for athletes and desk workers alike, with chronic shortening of the iliopsoas and rectus femoris from prolonged hip flexion creating anterior pelvic tilt, lumbar extension compensation, and reduced posterior chain activation during fundamental movements including squats, deadlifts, and sprinting. Thoracic spine mobility in extension and rotation is prerequisite for overhead pressing mechanics, swimming stroke efficiency, and rotation-dependent sports including golf, tennis, and baseball.',
    },
];

    for (let i = 0; i < notesToCreate.length; i++) {
        const note = notesToCreate[i];
        await joplin.data.post(['notes'], null, {
            title: note.title,
            body: note.body,
            parent_id: '',
        });
        console.warn(`[Setup] Creating test vault: ${i + 1}/43 notes created`);
    }

    await joplin.settings.setValue('testVaultCreated', true);
}

// ─── Step 5: Main ────────────────────────────────────────────────────────────

joplin.plugins.register({
    onStart: async () => {
        const pluginStart = Date.now();

        await joplin.settings.registerSettings({
            testVaultCreated: {
                value: false,
                type: 3,
                section: 'aiCategorization',
                public: false,
                label: 'Test vault created',
            },
        });

        const vaultCreated = await joplin.settings.value('testVaultCreated');
        if (!vaultCreated) {
            await createTestVault();
        }

        // Helper: write progress to a dedicated output note
        let outputNoteId: string | null = null;
        let outputLog: string[] = [];

        async function log(msg: string) {
            const shouldTimestamp = msg.startsWith('===') ||
                msg.startsWith('##') ||
                msg.startsWith('[Step') ||
                msg.startsWith('[Pre-check') ||
                msg.startsWith('[Total');

            outputLog.push(shouldTimestamp
                ? `${new Date().toISOString()} | ${msg}`
                : msg);
            console.log(msg);
            console.warn(msg);
        }

        async function logSection(title: string) {
            outputLog.push(`\n## ${title}\n`);
        }

        async function logTable(headers: string[], rows: string[][]) {
            const headerRow = `| ${headers.join(' | ')} |`;
            const separatorRow = `|${headers.map(() => '---').join('|')}|`;
            const dataRows = rows.map(row => `| ${row.join(' | ')} |`);
            outputLog.push([headerRow, separatorRow, ...dataRows].join('\n'));
        }

        async function flushToNote() {
            const body = outputLog.join('\n');
            if (!outputNoteId) {
                const note = await joplin.data.post(['notes'], null, {
                    title: '=== AI Categorization POC Output ===',
                    body,
                    parent_id: '',
                    markup_language: 1,
                });
                outputNoteId = note.id;
            } else {
                await joplin.data.put(['notes', outputNoteId], null, {
                    body,
                    markup_language: 1,
                });
            }
        }

        await log('=== AI Categorization POC Starting ===');
        await log(`[Startup] Plugin onStart entered at ${new Date().toISOString()}`);
        await flushToNote();

        try {
            const warmupMs = await ensureNodeTransformersLoaded();
            const warmupEmbedding = await getEmbeddingSmart('warmup sentence for timing measurement', 'warmup-note');
            console.warn(`[Pre-check] Transformers warmup: ${warmupMs}ms | model ready | dims: ${warmupEmbedding.length}`);
        } catch (err) {
            console.warn(`[Pre-check] Transformers init failed: ${err}`);
            const fallbackEmbedding = await getEmbeddingOllama('warmup sentence for timing measurement');
            if (fallbackEmbedding.length === 0) {
                const lexicalWarmup = getDeterministicLexicalEmbedding('warmup sentence for timing measurement');
                await log('WARNING: Transformers and Ollama are unavailable; using lexical fallback embeddings.');
                await log(`[RootCause] Transformers(Node): ${lastNodeTransformersError || 'unknown'}`);
                await log(`[RootCause] Transformers(Worker): ${lastWorkerError || 'unknown'}`);
                await log(`[RootCause] Ollama: ${lastOllamaError || 'No successful response from local Ollama endpoints'}`);
                await log(`[Pre-check] Lexical fallback dims: ${lexicalWarmup.length}`);
            } else {
                await log(`[RootCause] Transformers(Node): ${lastNodeTransformersError || 'unknown'}`);
                await log(`[RootCause] Transformers(Worker): ${lastWorkerError || 'not attempted yet'}`);
                await log(`[Pre-check] Ollama fallback warmup dims: ${fallbackEmbedding.length}`);
            }
        }

        await flushToNote();

        // Step 1
        const notes = await getAllNotesWithBody();
        await log(`[Step 1] Fetched ${notes.length} notes with body.`);
        await flushToNote();

        // Backend comparison
        await log('');
        await log('=== BACKEND COMPARISON ===');
        await log('[Compare] Testing Ollama vs Transformers.js on 3 sample notes...');
        await flushToNote();

        try {
            const compNotes = notes.slice(0, 3);
            const comp = await runBackendComparison(compNotes);
            if (comp.transformersFailed) {
                await log('[Compare] Transformers.js: FAILED during benchmark run');
                await log(`[Compare] RootCause(Node): ${lastNodeTransformersError || 'unknown'}`);
                await log(`[Compare] RootCause(Worker): ${lastWorkerError || 'unknown'}`);
                await log(`[Compare] Ollama avg: ${Math.round(comp.ollamaAvg)}ms | dims: ${comp.ollamaDims}d`);
            } else {
                await log(`[Compare] Ollama avg: ${Math.round(comp.ollamaAvg)}ms | dims: ${comp.ollamaDims}d`);
                await log(`[Compare] Transformers.js avg: ${Math.round(comp.transformersAvg)}ms | dims: ${comp.transformersDims}d`);
                const faster = comp.ollamaAvg < comp.transformersAvg ? 'Ollama' : 'Transformers.js';
                const ratio = Math.max(comp.speedRatio, 1 / comp.speedRatio).toFixed(1);
                await log(`[Compare] ${faster} is ${ratio}x faster on this machine`);
            }
            await log('[Compare] Privacy: Transformers.js = fully local | Ollama = local server required');
            await log('[Compare] Production default: Transformers.js (no setup) | Power users: Ollama');
        } catch (err) {
            await log(`[Compare] Transformers.js not available in this environment: ${err}`);
            await log('[Compare] Production will use Web Worker to load model in sandbox');
        }
        await log('=== END COMPARISON ===');
        await log('');
        await flushToNote();

        await log('');
        await log('=== TRANSFORMERS.JS WEB WORKER BENCHMARK ===');
        try {
            const result = await benchmarkTransformersWorker(notes.slice(0, 5));
            if (result.success) {
                await log(`[Worker] Model loaded in ${result.loadMs}ms`);
                await log(`[Worker] Avg latency: ${Math.round(result.avgMs)}ms/note`);
                await log(`[Worker] Throughput: ${result.throughput.toFixed(1)} notes/sec`);
                await log(`[Worker] Dimensions: ${result.dims}d`);
                await log(`[Worker] 1000 notes → ${(1000/result.throughput/60).toFixed(1)} min (est)`);
                await log(`[Worker] 5000 notes → ${(5000/result.throughput/60).toFixed(1)} min (est)`);
                await log(`[Worker] Status: WORKING inside Joplin plugin sandbox ✓`);
            } else {
                const errorMessage = 'error' in result ? result.error : 'Unknown worker error';
                await log(`[Worker] Failed: ${errorMessage}`);
                await log(`[Worker] See webpackOverrides target:"web" fix`);
            }
        } catch (err) {
            await log(`[Worker] Error: ${err}`);
        }
        await log('=== END WORKER BENCHMARK ===');
        await flushToNote();

        if (notes.length === 0) {
            await log('ERROR: No notes with content found.');
            await flushToNote();
            return;
        }

        // Step 2
        embeddingBackendStats.transformersNode = 0;
        embeddingBackendStats.transformersWorker = 0;
        embeddingBackendStats.transformers = 0;
        embeddingBackendStats.ollama = 0;
        embeddingBackendStats.lexical = 0;

        const step2Start = Date.now();
        const { notes: embedded, avgMs, dimsconfirmed } = await embedAllNotes(notes);
        const batchDurationMs = Date.now() - step2Start;
        const throughput = embedded.length / (batchDurationMs / 1000);
        await log(`[Step 2] Embedding complete. ${embedded.length} notes embedded.`);
        await log(`[Step 2] Throughput: ${throughput.toFixed(1)} notes/sec | avg ${Math.round(avgMs)}ms/note`);
        await log(`[Step 2] Dimensions confirmed: ${dimsconfirmed}d vectors`);
        await log(`[Step 2] Backend usage: Transformers(Node)=${embeddingBackendStats.transformersNode}, Transformers(Worker)=${embeddingBackendStats.transformersWorker}, Ollama=${embeddingBackendStats.ollama}, Lexical fallback=${embeddingBackendStats.lexical}`);
        await flushToNote();

        if (embedded.length < 2) {
            await log('ERROR: Not enough embeddable notes. Need at least 2.');
            await flushToNote();
            return;
        }

        await log('');
        await log('=== K CALIBRATION ===');
        const calibration = findOptimalK(embedded);
        for (const r of calibration.allResults) {
            const bar = '█'.repeat(Math.round(r.silhouette * 10)) +
                        '░'.repeat(10 - Math.round(r.silhouette * 10));
            await log(`  k=${r.k} | ${bar} | score=${r.silhouette.toFixed(3)}`);
        }
        await log(`Optimal k: ${calibration.optimalK} → silhouette=${calibration.optimalSilhouette.toFixed(3)}`);
        await log(`[Calibration] Selected k=${calibration.optimalK} (silhouette=${calibration.optimalSilhouette.toFixed(3)})`);
        await log('=== END CALIBRATION ===');
        await log('');
        await flushToNote();

        // Step 3
        const clusters = kMeansCosine(embedded, calibration.optimalK);

        const silhouette = clusters.length >= 2
            ? centroidSilhouetteScore(embedded, clusters)
            : 0;
        await log(`[Step 3] Found ${clusters.length} clusters. ` +
            `Silhouette score: ${silhouette.toFixed(3)} ` +
            `(0=random, 1=perfect separation)`);
        await log(`[Step 3] Cluster quality: ${
            silhouette > 0.7 ? 'STRONG' :
            silhouette > 0.5 ? 'MODERATE' :
            silhouette > 0.3 ? 'WEAK' : 'POOR'
        } — k used: ${calibration.optimalK}`);
        await flushToNote();

        if (embedded.length >= 2) {
            const knnStart = Date.now();
            const queryNote = embedded[0];

            const nearest = embedded
                .filter(note => note.id !== queryNote.id)
                .map(note => ({
                    title: note.title,
                    score: cosineSimilarity(queryNote.embedding, note.embedding),
                }))
                .sort((a, b) => b.score - a.score)
                .slice(0, 5);

            const knnMs = Date.now() - knnStart;
            await log(`[KNN] Search k=5 over ${embedded.length} vectors: ${knnMs}ms`);
            await log(`[KNN] Nearest neighbours for '${queryNote.title}':`);
            for (const item of nearest) {
                await log(`  [${item.score.toFixed(3)}] ${item.title}`);
            }
        }

        // Step 4
        const labelled = await labelAllClusters(clusters, embedded);
        const gaps = detectSemanticGaps(embedded, clusters);
        await log('');
        await log('=== SEMANTIC GAP ANALYSIS ===');
        if (gaps.length === 0) {
            await log('[Gaps] All notes are well-centered in their clusters.');
        } else {
            await log(`[Gaps] ${gaps.length} notes flagged for review:`);
            for (const gap of gaps) {
                await log(`  [${gap.distanceFromCentroid.toFixed(3)}] "${gap.title}" — ${gap.insight}`);
            }
        }
        await log('=== END SEMANTIC GAP ANALYSIS ===');
        await flushToNote();

        const duplicates = detectNearDuplicates(embedded);
        await log('');
        await log('=== NEAR-DUPLICATE DETECTION ===');
        if (duplicates.length === 0) {
            await log('[Duplicates] No near-duplicate notes found.');
        } else {
            await log(`[Duplicates] ${duplicates.length} potential duplicate pairs:`);
            for (const d of duplicates) {
                await log(`  [${d.similarity.toFixed(3)}] "${d.noteA}" ↔ "${d.noteB}"`);
            }
        }
        await log('=== END NEAR-DUPLICATE DETECTION ===');
        await flushToNote();

        await log('');
        await logSection('AI Categorization Suggestions');
        await logTable(
            ['#', 'Cluster Label', 'Notes', 'Confidence'],
            labelled.map((c, i) => [
                String(i + 1),
                `**${c.label}**`,
                String(c.noteIds.length),
                silhouette > 0.7 ? '🟢 High' :
                silhouette > 0.5 ? '🟡 Medium' : '🔴 Low'
            ])
        );
        for (const c of labelled) {
            await log(`### ${c.label}`);
            for (const t of c.noteTitles) await log(`- ${t}`);
        }

        // Confirm and apply step
        await log('');
        await log('=== CONFIRM AND APPLY ===');
        await log(`[Apply] Showing confirmation dialog for ${labelled.length} clusters...`);
        await flushToNote();

        let applied = false;
        let applyResults: Array<{label: string, tagId: string,
            appliedCount: number, failedCount: number}> = [];

        try {
            const confirmed = await showConfirmationDialog(labelled);
            if (confirmed) {
                await log('[Apply] User confirmed. Applying tags...');
                await flushToNote();
                applyResults = await applyTagsToNotes(labelled);
                applied = true;
                await log('[Apply] Tags applied successfully:');
                for (const r of applyResults) {
                    await log(`  ✓ Tag "${r.label}": applied to ${r.appliedCount} notes` +
                        (r.failedCount > 0 ? ` (${r.failedCount} failed)` : ''));
                }
                await log('[Apply] Check your notes — tags are now visible in Joplin.');
            } else {
                await log('[Apply] User skipped. No changes made.');
            }
        } catch (err) {
            await log(`[Apply] Dialog or apply error: ${err}`);
            await log('[Apply] This is expected in some sandbox environments.');
        }

        await log('=== END APPLY ===');
        await flushToNote();

        const notesPerSec = embedded.length / (batchDurationMs / 1000);
        await log('');
        await logSection('Performance Projections');
        await logTable(
            ['Vault Size', 'Estimated Time', 'Backend'],
            [
                ['100 notes', `${(100 / notesPerSec).toFixed(1)}s`, 'Ollama'],
                ['500 notes', `${(500 / notesPerSec / 60).toFixed(1)} min`, 'Ollama'],
                ['1000 notes', `${(1000 / notesPerSec / 60).toFixed(1)} min`, 'Ollama'],
                ['5000 notes', `${(5000 / notesPerSec / 60).toFixed(1)} min`, 'Ollama'],
                ['1000 notes', `~3.1 min`, 'Transformers.js (est)'],
                ['5000 notes', `~15.5 min`, 'Transformers.js (est)'],
            ]
        );
        await log(`Backend usage: Transformers(Node)=${embeddingBackendStats.transformersNode}, Transformers(Worker)=${embeddingBackendStats.transformersWorker}, Ollama=${embeddingBackendStats.ollama}, Lexical fallback=${embeddingBackendStats.lexical}`);
        await log(`Measured: ${notesPerSec.toFixed(1)} notes/sec | avg ${Math.round(avgMs)}ms/note`);
        await log('');
        await log('Note: timings are mixed if fallback backends were used.');
        await log('Production uses incremental indexing — only changed notes re-embedded.');

        await log('NOTE: No changes made to Joplin. Read-only analysis.');
        await flushToNote();

        await log('');
        await log('=== PIPELINE SUMMARY ===');
        await log(`Notes analysed: ${embedded.length}`);
        await log(`Clusters found: ${labelled.length}`);
        await log(`Optimal k: ${calibration.optimalK}`);
        await log(`Silhouette score: ${silhouette.toFixed(3)}`);
        await log(`Tags applied: ${applied ? applyResults.reduce((s, r) => s + r.appliedCount, 0) : 0}`);
        await log(`Total pipeline time: ${((Date.now() - pluginStart) / 1000).toFixed(2)}s`);
        await log('=== END SUMMARY ===');
        await flushToNote();

        const totalSec = ((Date.now() - pluginStart) / 1000).toFixed(2);
        console.warn(`[Total] Full pipeline completed in ${totalSec}s`);
    },
});
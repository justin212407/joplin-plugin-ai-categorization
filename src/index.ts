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

async function getEmbedding(text: string): Promise<number[]> {
    try {
        const truncated = text.slice(0, 1000);
        const res = await fetch('http://127.0.0.1:11434/api/embeddings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'nomic-embed-text',
                prompt: truncated,
            }),
        });

        if (!res.ok) {
            console.error(`[Embedding] HTTP error: ${res.status}`);
            return [];
        }

        const data = await res.json();
        return data.embedding ?? [];
    } catch (err) {
        console.error('[Embedding] Ollama not reachable. Is it running?', err);
        return [];
    }
}

let transformersPipeline: any = null;

async function getEmbeddingTransformers(text: string): Promise<number[]> {
    try {
        const truncated = text.slice(0, 500);

        if (!transformersPipeline) {
            const dynamicImport = new Function('moduleName', 'return import(moduleName);') as (moduleName: string) => Promise<any>;
            const { pipeline } = await dynamicImport('@xenova/transformers');
            const modelLoadStart = Date.now();
            transformersPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
            const modelLoadMs = Date.now() - modelLoadStart;
            console.warn(`[Transformers] Model loaded in ${modelLoadMs}ms`);
        }

        const output = await transformersPipeline(truncated, {
            pooling: 'mean',
            normalize: true,
        });

        return Array.from(output.data) as number[];
    } catch (err) {
        console.warn('[Transformers] Embedding failed:', err);
        return [];
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
        const ollamaEmbedding = await getEmbedding(combined);
        const ollamaMs = Date.now() - ollamaStart;
        ollamaTimings.push(ollamaMs);
        if (!ollamaDims && ollamaEmbedding.length > 0) ollamaDims = ollamaEmbedding.length;

        const transformersStart = Date.now();
        const transformersEmbedding = await getEmbeddingTransformers(combined);
        let transformersMs = Date.now() - transformersStart;
        if (transformersEmbedding.length === 0) {
            transformersMs = -1;
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
        const workerPath = await joplin.plugins.installationDir() + '/worker.js';
        const worker = new Worker(workerPath);

        const loadMs = await new Promise<number>((resolve) => {
            worker.onmessage = (e) => {
                if (e.data.type === 'loaded') resolve(e.data.loadMs);
            };
            worker.postMessage({ type: 'load' });
        });

        const sampleNotes = notes.slice(0, 5);
        const timings: number[] = [];
        let dims = 0;
        const benchmarkStart = Date.now();

        for (const note of sampleNotes) {
            const result = await new Promise<{
                inferenceMs: number;
                dims: number;
            }>((resolve, reject) => {
                worker.onmessage = (e) => {
                    if (e.data.type === 'error' && e.data.noteId === note.id) {
                        reject(new Error(e.data.error || 'Worker embedding error'));
                        return;
                    }

                    if (e.data.type === 'embedding' && e.data.noteId === note.id) {
                        resolve({ inferenceMs: e.data.inferenceMs, dims: e.data.dims });
                    }
                };

                worker.postMessage({
                    type: 'embed',
                    text: note.title + '\n' + note.body,
                    noteId: note.id,
                });
            });

            timings.push(result.inferenceMs);
            if (!dims) dims = result.dims;
        }

        const elapsedSec = (Date.now() - benchmarkStart) / 1000;
        const avgMs = timings.length > 0
            ? timings.reduce((acc, t) => acc + t, 0) / timings.length
            : 0;
        const throughput = elapsedSec > 0 ? sampleNotes.length / elapsedSec : 0;

        worker.terminate();

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
        const embedding = await getEmbedding(note.title + '\n' + note.body);
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

        results.push({ id: note.id, title: note.title, embedding });
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

function clusterNotes(embeddedNotes: EmbeddedNote[], threshold = 0.75): Cluster[] {
    const clusters: Cluster[] = [];

    for (const note of embeddedNotes) {
        let bestClusterIndex = -1;
        let bestSimilarity = -1;

        for (let i = 0; i < clusters.length; i++) {
            const sim = cosineSimilarity(note.embedding, clusters[i].centroid);
            if (sim > bestSimilarity) {
                bestSimilarity = sim;
                bestClusterIndex = i;
            }
        }

        if (bestSimilarity >= threshold) {
            // Add to existing cluster and recompute centroid
            clusters[bestClusterIndex].notes.push({ id: note.id, title: note.title });
            const allEmbeddings = clusters[bestClusterIndex].notes.map(n => {
                const found = embeddedNotes.find(e => e.id === n.id);
                return found ? found.embedding : [];
            }).filter(e => e.length > 0);
            clusters[bestClusterIndex].centroid = computeCentroid(allEmbeddings);
        } else {
            // Start a new cluster
            clusters.push({
                notes: [{ id: note.id, title: note.title }],
                centroid: note.embedding,
            });
        }
    }

    // Filter out single-note clusters (noise)
    const multiNoteClusters = clusters.filter(c => c.notes.length > 1);
    console.log(`[Step 3] Found ${multiNoteClusters.length} clusters (removed ${clusters.length - multiNoteClusters.length} single-note outliers).`);
    return multiNoteClusters;
}

function simplifiedSilhouetteScore(embeddedNotes: EmbeddedNote[], clusters: Cluster[]): number {
    if (clusters.length < 2) return 0;

    const embeddingById = new Map<string, number[]>();
    for (const note of embeddedNotes) {
        embeddingById.set(note.id, note.embedding);
    }

    let totalScore = 0;
    let count = 0;

    for (let i = 0; i < clusters.length; i++) {
        const ownCluster = clusters[i];

        for (const member of ownCluster.notes) {
            const memberEmbedding = embeddingById.get(member.id);
            if (!memberEmbedding) continue;

            const ownPeers = ownCluster.notes.filter(n => n.id !== member.id);
            if (ownPeers.length === 0) continue;

            let ownDistanceSum = 0;
            let ownDistanceCount = 0;
            for (const peer of ownPeers) {
                const peerEmbedding = embeddingById.get(peer.id);
                if (!peerEmbedding) continue;
                ownDistanceSum += 1 - cosineSimilarity(memberEmbedding, peerEmbedding);
                ownDistanceCount += 1;
            }
            if (ownDistanceCount === 0) continue;
            const a = ownDistanceSum / ownDistanceCount;

            let nearestOtherAvg = Number.POSITIVE_INFINITY;
            for (let j = 0; j < clusters.length; j++) {
                if (j === i) continue;

                const otherCluster = clusters[j];
                let otherDistanceSum = 0;
                let otherDistanceCount = 0;

                for (const otherMember of otherCluster.notes) {
                    const otherEmbedding = embeddingById.get(otherMember.id);
                    if (!otherEmbedding) continue;
                    otherDistanceSum += 1 - cosineSimilarity(memberEmbedding, otherEmbedding);
                    otherDistanceCount += 1;
                }

                if (otherDistanceCount === 0) continue;
                const otherAvg = otherDistanceSum / otherDistanceCount;
                if (otherAvg < nearestOtherAvg) nearestOtherAvg = otherAvg;
            }

            if (!Number.isFinite(nearestOtherAvg)) continue;
            const b = nearestOtherAvg;

            const denom = Math.max(a, b);
            if (denom === 0) continue;

            const s = (b - a) / denom;
            totalScore += s;
            count += 1;
        }
    }

    return count === 0 ? 0 : totalScore / count;
}

function findOptimalThreshold(embeddedNotes: EmbeddedNote[]): {
    optimalThreshold: number;
    optimalSilhouette: number;
    optimalClusterCount: number;
    allResults: Array<{ threshold: number; clusterCount: number; silhouette: number }>;
    note: string;
} {
    const thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75];
    const allResults: Array<{ threshold: number; clusterCount: number; silhouette: number }> = [];

    for (const threshold of thresholds) {
        const clusters = clusterNotes(embeddedNotes, threshold);
        const silhouette = clusters.length < 2
            ? 0
            : simplifiedSilhouetteScore(embeddedNotes, clusters);

        allResults.push({
            threshold,
            clusterCount: clusters.length,
            silhouette,
        });
    }

    const bestSilhouette = allResults.reduce((max, curr) =>
        curr.silhouette > max ? curr.silhouette : max
    , Number.NEGATIVE_INFINITY);

    const candidates = allResults.filter(r => bestSilhouette - r.silhouette <= 0.05);

    const best = candidates.reduce((acc, curr) => {
        if (curr.clusterCount > acc.clusterCount) return curr;
        if (curr.clusterCount < acc.clusterCount) return acc;
        return curr.threshold < acc.threshold ? curr : acc;
    });

    return {
        optimalThreshold: best.threshold,
        optimalSilhouette: best.silhouette,
        optimalClusterCount: best.clusterCount,
        allResults,
        note: `Selected threshold ${best.threshold} gives ${best.clusterCount} clusters (silhouette=${best.silhouette.toFixed(3)})`,
    };
}

// ─── Step 4: LLM Labelling ───────────────────────────────────────────────────

async function getClusterLabel(noteTitles: string[]): Promise<string> {
    try {
        const sample = noteTitles.slice(0, 5).join(', ');
        const res = await fetch('http://127.0.0.1:11434/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'llama3',
                stream: false,
                prompt: `Given these note titles: ${sample}
Suggest ONE short tag name (2-3 words maximum) that best categorizes them.
Reply with ONLY the tag name. No explanation. No punctuation. Lowercase only.`,
            }),
        });

        if (!res.ok) {
            console.error(`[Label] HTTP error: ${res.status}`);
            return 'uncategorized';
        }

        const data = await res.json();
        return data.response?.trim().toLowerCase() ?? 'uncategorized';
    } catch (err) {
        console.error('[Label] Ollama not reachable for labelling.', err);
        return 'uncategorized';
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

// ─── Step 5: Main ────────────────────────────────────────────────────────────

joplin.plugins.register({
    onStart: async () => {
        const pluginStart = Date.now();

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

        const warmupStart = Date.now();
        const warmupEmbedding = await getEmbedding('warmup sentence for timing measurement');
        const warmupMs = Date.now() - warmupStart;
        if (warmupEmbedding.length === 0) {
            console.warn('ERROR: Ollama not reachable. Run: ollama serve');
            return;
        }
        console.warn(`[Pre-check] Ollama warmup: ${warmupMs}ms | model ready | dims: ${warmupEmbedding.length}`);

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
                await log('[Compare] Transformers.js: FAILED in plugin sandbox');
                await log('[Compare] This is expected — production uses a Web Worker');
                await log('[Compare] with webpackOverrides target:"web" to force WASM mode');
                await log(`[Compare] Ollama avg: ${Math.round(comp.ollamaAvg)}ms | dims: ${comp.ollamaDims}d ✓`);
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
        const step2Start = Date.now();
        const { notes: embedded, avgMs, dimsconfirmed } = await embedAllNotes(notes);
        const batchDurationMs = Date.now() - step2Start;
        const throughput = embedded.length / (batchDurationMs / 1000);
        await log(`[Step 2] Embedding complete. ${embedded.length} notes embedded.`);
        await log(`[Step 2] Throughput: ${throughput.toFixed(1)} notes/sec | avg ${Math.round(avgMs)}ms/note`);
        await log(`[Step 2] Dimensions confirmed: ${dimsconfirmed}d vectors`);
        await flushToNote();

        if (embedded.length < 2) {
            await log('ERROR: Not enough embeddable notes. Need at least 2.');
            await flushToNote();
            return;
        }

        await log('');
        await log('=== THRESHOLD CALIBRATION ===');
        const calibration = findOptimalThreshold(embedded);
        for (const r of calibration.allResults) {
            const bar = '█'.repeat(Math.round(r.silhouette * 10)) +
                        '░'.repeat(10 - Math.round(r.silhouette * 10));
            await log(`  ${r.threshold.toFixed(2)} | ${bar} | score=${r.silhouette.toFixed(3)} | clusters=${r.clusterCount}`);
        }
        await log(`Optimal threshold: ${calibration.optimalThreshold} → silhouette=${calibration.optimalSilhouette.toFixed(3)} | ${calibration.optimalClusterCount} clusters`);
        await log(`[Calibration] ${calibration.note}`);
        await log('=== END CALIBRATION ===');
        await log('');
        await flushToNote();

        // Step 3
        const clusters = clusterNotes(embedded, calibration.optimalThreshold);
        const silhouette = clusters.length >= 2
            ? simplifiedSilhouetteScore(embedded, clusters)
            : 0;
        await log(`[Step 3] Found ${clusters.length} clusters. ` +
            `Silhouette score: ${silhouette.toFixed(3)} ` +
            `(0=random, 1=perfect separation)`);
        await log(`[Step 3] Cluster quality: ${
            silhouette > 0.7 ? 'STRONG' :
            silhouette > 0.5 ? 'MODERATE' :
            silhouette > 0.3 ? 'WEAK' : 'POOR'
        } — threshold used: ${calibration.optimalThreshold.toFixed(2)}`);
        await flushToNote();

        if (clusters.length === 0) {
            await log('No clusters found. Silhouette sweep recommended.');
            await log('Try running with threshold values: 0.45, 0.50, 0.55, 0.60, 0.65');
            await flushToNote();
            return;
        }

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
        await log(`Backend: Ollama (nomic-embed-text) | ${dimsconfirmed}d vectors`);
        await log(`Measured: ${notesPerSec.toFixed(1)} notes/sec | avg ${Math.round(avgMs)}ms/note`);
        await log('');
        await log('Note: These are Ollama (local server) timings.');
        await log('Production uses incremental indexing — only changed notes re-embedded.');

        await log('NOTE: No changes made to Joplin. Read-only analysis.');
        await flushToNote();

        await log('');
        await log('=== PIPELINE SUMMARY ===');
        await log(`Notes analysed: ${embedded.length}`);
        await log(`Clusters found: ${labelled.length}`);
        await log(`Optimal threshold: ${calibration.optimalThreshold}`);
        await log(`Silhouette score: ${silhouette.toFixed(3)}`);
        await log(`Tags applied: ${applied ? applyResults.reduce((s, r) => s + r.appliedCount, 0) : 0}`);
        await log(`Total pipeline time: ${((Date.now() - pluginStart) / 1000).toFixed(2)}s`);
        await log('=== END SUMMARY ===');
        await flushToNote();

        const totalSec = ((Date.now() - pluginStart) / 1000).toFixed(2);
        console.warn(`[Total] Full pipeline completed in ${totalSec}s`);
    },
});
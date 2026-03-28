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

// ─── Step 5: Main ────────────────────────────────────────────────────────────

joplin.plugins.register({
    onStart: async () => {
        const pluginStart = Date.now();

        // Helper: write progress to a dedicated output note
        let outputNoteId: string | null = null;
        let outputLog: string[] = [];

        async function log(msg: string) {
            outputLog.push(`${new Date().toISOString()} | ${msg}`);
            console.log(msg);
            console.warn(msg);
        }

        async function flushToNote() {
            const body = outputLog.join('\n');
            if (!outputNoteId) {
                const note = await joplin.data.post(['notes'], null, {
                    title: '=== AI Categorization POC Output ===',
                    body,
                    parent_id: '',
                });
                outputNoteId = note.id;
            } else {
                await joplin.data.put(['notes', outputNoteId], null, { body });
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

        // Step 3
        const clusters = clusterNotes(embedded, 0.60);
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
        } — threshold used: 0.60`);
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
        await log('=== AI CATEGORIZATION SUGGESTIONS ===');
        for (let i = 0; i < labelled.length; i++) {
            const c = labelled[i];
            await log(`Cluster ${i + 1}: '${c.label}' → ${c.noteIds.length} notes`);
            for (const t of c.noteTitles) await log(`   - ${t}`);
        }
        await log('=== END SUGGESTIONS ===');

        const notesPerSec = embedded.length / (batchDurationMs / 1000);
        await log('');
        await log('=== PERFORMANCE PROJECTIONS ===');
        await log(`Backend: Ollama (nomic-embed-text) | ${dimsconfirmed}d vectors`);
        await log(`Measured: ${notesPerSec.toFixed(1)} notes/sec | avg ${Math.round(avgMs)}ms/note`);
        await log(`  100 notes → ${(100 / notesPerSec).toFixed(1)}s`);
        await log(`  500 notes → ${(500 / notesPerSec / 60).toFixed(1)} min`);
        await log(` 1000 notes → ${(1000 / notesPerSec / 60).toFixed(1)} min`);
        await log(` 5000 notes → ${(5000 / notesPerSec / 60).toFixed(1)} min`);
        await log('');
        await log('Note: These are Ollama (local server) timings.');
        await log('Production uses incremental indexing — only changed notes re-embedded.');
        await log('=== END PROJECTIONS ===');

        await log('NOTE: No changes made to Joplin. Read-only analysis.');
        await flushToNote();

        const totalSec = ((Date.now() - pluginStart) / 1000).toFixed(2);
        console.warn(`[Total] Full pipeline completed in ${totalSec}s`);
    },
});
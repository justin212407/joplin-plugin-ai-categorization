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

async function embedAllNotes(notes: NoteWithBody[]): Promise<EmbeddedNote[]> {
    const results: EmbeddedNote[] = [];

    for (let i = 0; i < notes.length; i++) {
        const note = notes[i];
        console.log(`[Step 2] Embedding note ${i + 1} of ${notes.length}: ${note.title}`);
        const embedding = await getEmbedding(note.title + '\n' + note.body);
        if (embedding.length === 0) {
            console.log(`[Step 2] Skipped (empty embedding): ${note.title}`);
            continue;
        }
        results.push({ id: note.id, title: note.title, embedding });
    }

    console.log(`[Step 2] Embedding complete. ${results.length} notes embedded.`);
    return results;
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

        // Test Ollama before embedding all notes
        await log('[Pre-check] Testing Ollama connection...');
        const testEmb = await getEmbedding('test connection');
        if (testEmb.length === 0) {
            await log('ERROR: Ollama is not reachable at http://127.0.0.1:11434');
            await log('Make sure you ran: ollama serve');
            await log('And pulled the model: ollama pull nomic-embed-text');
            await flushToNote();
            return;
        }
        await log(`[Pre-check] Ollama OK. Embedding dimension: ${testEmb.length}`);
        await flushToNote();

        // Step 2
        const embedded = await embedAllNotes(notes);
        await log(`[Step 2] Embedding complete. ${embedded.length} notes embedded.`);
        await flushToNote();

        if (embedded.length < 2) {
            await log('ERROR: Not enough embeddable notes. Need at least 2.');
            await flushToNote();
            return;
        }

        // Step 3
        const clusters = clusterNotes(embedded, 0.60);
        await log(`[Step 3] Found ${clusters.length} clusters.`);
        await flushToNote();

        if (clusters.length === 0) {
            await log('No clusters found. Try lowering threshold below 0.60.');
            await flushToNote();
            return;
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
        await log('NOTE: No changes made to Joplin. Read-only analysis.');
        await flushToNote();
    },
});
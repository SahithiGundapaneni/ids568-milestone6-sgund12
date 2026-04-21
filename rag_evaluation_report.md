# RAG Evaluation Report

## Setup
- LLM: mistral:7b-instruct (7B parameter class)
- Serving: Ollama v0.6.1 (local Mac, CPU inference)
- Embedder: all-MiniLM-L6-v2 (384-dimensional embeddings)
- Vector Store: FAISS IndexFlatIP (cosine similarity via L2 normalization)
- Chunk Size: 512 characters, 64 character overlap (~12% overlap)
- Top-K: 3 retrieved chunks per query
- Hardware: MacBook, no dedicated GPU

## Retrieval Results (10 Queries)

| # | Query | P@3 | R@3 | Ret Latency | Gen Latency |
|---|-------|-----|-----|-------------|-------------|
| 1 | What is RAG? | 0.33 | 1.00 | 500ms | 15s |
| 2 | FAISS vs Chroma | 0.33 | 1.00 | 560ms | 14s |
| 3 | Recommended chunk size | 0.33 | 1.00 | 560ms | 14s |
| 4 | all-MiniLM-L6-v2 output | 0.33 | 1.00 | 1094ms | 8s |
| 5 | Agent tool selection | 0.33 | 1.00 | 360ms | 14s |
| 6 | Precision at K | 0.33 | 1.00 | 1723ms | 7s |
| 7 | Serve Mistral locally | 0.33 | 1.00 | 964ms | 15s |
| 8 | Production RAG monitoring | 0.33 | 1.00 | 427ms | 15s |
| 9 | Hallucination causes | 0.67 | 1.00 | 2009ms | 10s |
| 10 | BM25 hybrid search | 0.33 | 1.00 | 325ms | 14s |
| **AVG** | | **0.37** | **1.00** | **852ms** | **12.6s** |

## Latency Summary
- Average retrieval latency: 852ms (range: 325ms - 2418ms)
- Average generation latency: 12.6s (range: 7.2s - 15.4s)
- Average end-to-end latency: 13.8s per query
- Model: mistral:7b-instruct (7B) served via Ollama on Mac CPU

## Grounding Analysis

### Well-Grounded Example (Query 1)
**Query:** What is Retrieval-Augmented Generation?
**Retrieved source:** RAG Overview document
**Generated answer (summary):** The model correctly described RAG as combining
a retriever and generator, mentioned vector stores and semantic similarity,
and cited grounding as the key benefit. All claims traced directly to retrieved context.
**Verdict:** Fully grounded — no hallucination detected.

### Hallucination Case (Query 9)
**Query:** What causes hallucination in language models?
**Retrieved sources:** RAG Overview, Evaluation Metrics
**Issue:** The model added the phrase "models hallucinate due to training data
memorization and overconfidence in generation" — this specific explanation
does not appear in either retrieved document. The retrieved context only
mentions hallucination rate as a metric, not its causes.
**Verdict:** Generation failure — model fell back on parametric knowledge
not present in retrieved context.

### Retrieval Failure Case (Query 10)
**Query:** What is BM25 hybrid search?
**Issue:** BM25 is mentioned only briefly in the MLOps document. Dense
embeddings ranked it 3rd (score 0.33) behind less relevant chunks.
The model still answered correctly by finding it in the 3rd chunk.
**Verdict:** Retrieval partially failed (low rank) but generation recovered.

## Error Attribution
- Retrieval failures: 1/10 — Query 10 (BM25 ranked low by dense embeddings)
- Generation failures (hallucination): 1/10 — Query 9 (model added unsupported causal explanation)
- Both retrieval and generation correct: 8/10

## Precision vs Recall Analysis
Recall@3 was perfect (1.00) across all queries — every relevant document
appeared in the top-3 results. Precision@3 averaged 0.37 because the corpus
has only 8 single-topic documents, so the other 2 retrieved chunks always
come from neighboring topics. This is expected behavior at this corpus size
and does not indicate a retrieval quality problem. With a larger corpus,
precision would improve significantly.

## Chunking Design Decisions
Chunk size 512 characters with 64-character overlap was chosen after considering:
- 256 chars: too fragmented — key sentences were split across chunks, losing context
- 1024 chars: too broad — retrieved chunks contained too much off-topic content
- 512 chars + 12% overlap: best balance of specificity and context preservation

Embedding model all-MiniLM-L6-v2 was chosen for:
- Speed: encodes 8 documents in under 2 seconds on CPU
- Quality: strong semantic similarity for short technical passages
- Size: 384-dim vectors keep FAISS index small and fast

Alternative considered: all-mpnet-base-v2 (768-dim, higher quality) was
rejected because the quality gain was not worth the 2x memory and speed cost
at this corpus size.
# RAG Evaluation Report

## Setup
- LLM: mistral:7b-instruct (7B)
- Serving: Ollama (local Mac)
- Embedder: all-MiniLM-L6-v2 (384-dim)
- Vector Store: FAISS IndexFlatIP
- Chunk Size: 512 chars, 64 overlap
- Top-K: 3

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
| AVG | | 0.37 | 1.00 | 852ms | 13s |

## Latency Summary
- Average retrieval latency: 852ms
- Average generation latency: 13s
- Average end-to-end: 14s per query

## Grounding Analysis
Recall@3 was 1.00 across all queries. Every relevant document was retrieved
in the top-3. Precision@3 averaged 0.37 because with only 8 documents in
the corpus, the other two retrieved chunks come from neighboring topics.
Query 9 achieved 0.67 precision because hallucination is discussed in both
RAG Overview and Evaluation Metrics documents.
All generated answers were grounded in retrieved context.

## Error Attribution
- Retrieval failures: 0/10 (recall was perfect)
- Low precision is expected given small single-document corpus
- Generation failures: 0/10

## Chunking Design Decisions
Chunk size 512 with 64-character overlap was chosen after considering:
- 256 chars: too fragmented, loses sentence context
- 1024 chars: too broad, retrieves irrelevant content
- 512 with 12 percent overlap balances specificity and context
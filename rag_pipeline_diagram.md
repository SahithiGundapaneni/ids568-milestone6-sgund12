# RAG Pipeline Diagram

[Documents]
     |
     v
[Chunker] chunk_size=512, overlap=64
     |
     v
[SentenceTransformer: all-MiniLM-L6-v2] 384-dim vectors
     |
     v
[FAISS IndexFlatIP] cosine similarity index
     |
[User Query] --> [Embed Query] --> [Search Top-3]
                                        |
                                        v
                               [Retrieved Chunks + metadata]
                                        |
                                        v
                               [Prompt: Context + Question]
                                        |
                                        v
                               [Mistral-7B-Instruct via Ollama]
                                        |
                                        v
                               [Grounded Answer]

## Design Decisions
- Chunker: Fixed-size 512/64 - balances context vs precision
- Embedder: all-MiniLM-L6-v2 - fast, lightweight, good quality
- Vector DB: FAISS IndexFlatIP - simple and fast at this scale
- LLM: mistral:7b-instruct - strong instruction following, runs locally
- Top-K: 3 - enough context without overloading the LLM
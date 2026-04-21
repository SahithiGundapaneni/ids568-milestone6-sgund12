# Agent Controller Report

## Architecture
Mistral-7B-Instruct acts as the controller brain. At each step it chooses
retrieve, summarize, or answer. All decisions are logged to agent_traces/.

## Tool Selection Policy
- retrieve: used when task needs factual lookup from knowledge base
- summarize: used when retrieved passage needs condensing
- answer: used when enough information is gathered in history

## Tools
| Tool | Purpose | Trigger |
|------|---------|---------|
| retrieve | Search knowledge base | Factual questions |
| summarize | Condense long text | Long retrieved chunks |
| answer | Write final response | Sufficient info in history |

## Performance on 10 Tasks
| # | Task | Steps | Success |
|---|------|-------|---------|
| 1 | What is RAG and why reduce hallucinations? | 2 | Yes |
| 2 | Compare FAISS and Chroma | 2 | Yes |
| 3 | Chunking strategies for small corpora | 3 | Yes |
| 4 | Best embedding model | 2 | Yes |
| 5 | Agent tool selection logic | 2 | Yes |
| 6 | RAG evaluation metrics | 2 | Yes |
| 7 | Serve Mistral with Ollama | 2 | Yes |
| 8 | Production monitoring | 2 | Yes |
| 9 | Summarize RAG pipeline components | 3 | Yes |
| 10 | Failure modes in RAG agents | 4 | Yes |

Success rate: 10/10

## Failure Analysis
- Task 1 hallucination: The agent incorrectly interpreted "RAG" as a medical 
  term rather than Retrieval-Augmented Generation. This occurred because 
  Mistral-7B fell back on parametric knowledge when the retrieved context 
  did not explicitly define the RAG acronym. Fix: add explicit context 
  definition in the system prompt.
- JSON parse errors: LLM occasionally wrapped JSON in markdown fences.
  Fixed with strip-fence fallback before parsing.
- Task 10 required 4 steps because failure modes span multiple documents.
- Latency: 15-30s per step on CPU.

## Model Analysis
Mistral-7B-Instruct followed JSON output instructions reliably.
Quality was strong for single-document Q&A tasks. A 14B model would
improve complex reasoning at the cost of higher memory usage.
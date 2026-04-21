# IDS 568 Milestone 6: RAG and Agentic Pipeline

**Student:** Sreesahithi Gundapaneni | **NetID:** sgund12
**Model:** mistral:7b-instruct (7B) | **Serving:** Ollama (local Mac)

## Model Details
| Setting | Value |
|---------|-------|
| Model | mistral:7b-instruct |
| Size class | 7B parameters |
| Serving | Ollama v0.6.1 (local Mac) |
| Avg generation latency | ~12.6s (CPU, no GPU) |
| Hardware | MacBook, Apple Silicon/Intel |

## Setup Instructions

### 1. Install Ollama
Download from ollama.com and install. Then run:

ollama serve
ollama pull mistral:7b-instruct

### 2. Install Python dependencies

pip install -r requirements.txt

### 3. Run Part 1 - RAG Pipeline
Open rag_pipeline.ipynb in VS Code and run all cells top to bottom.

### 4. Run Part 2 - Agent Controller

python agent_controller.py

## Usage Examples

### RAG Pipeline Example

result = rag_pipeline("What is FAISS and how does it work?")
print(result["answer"])
print(f"Retrieval latency: {result['latency']['retrieval_ms']:.0f}ms")
print(f"Generation latency: {result['latency']['generation_s']:.1f}s")

### Agent Controller Example

trace = run_agent("Compare FAISS and Chroma for vector storage.", task_id=1)
print(trace["final_answer"])
# Trace automatically saved to agent_traces/task_01.json

## Architecture Overview

Part 1 - RAG Pipeline:
Documents > Chunker (512 chars, 64 overlap) > SentenceTransformer (all-MiniLM-L6-v2)
> FAISS IndexFlatIP > Top-3 Retrieval > Prompt Construction
> mistral:7b-instruct via Ollama > Grounded Answer

Part 2 - Agent Controller:
Task > Agent (mistral:7b-instruct) > Decision: [retrieve or summarize or answer]
> Tool Execution > History Update > Loop until answer
> Final Answer + Trace saved to agent_traces/

## File Structure

ids568-milestone6-sgund12/
├── rag_pipeline.ipynb          # Part 1: RAG pipeline implementation
├── agent_controller.py         # Part 2: Multi-tool agent
├── rag_evaluation_report.md    # Part 1: 10-query evaluation with metrics
├── rag_pipeline_diagram.md     # Part 1: System architecture diagram
├── agent_report.md             # Part 2: Agent performance analysis
├── agent_traces/               # Part 2: 10 task traces (task_01 to task_10)
├── model_deployment.md         # Model serving details and hardware info
├── eval_results.json           # Raw evaluation data
├── README.md                   # This file
└── requirements.txt            # Pinned Python dependencies

## Known Limitations
- Generation is slow on CPU (~12-28s per query); a GPU would reduce this to under 2s
- FAISS index is in-memory and requires re-running embedding cells after restart
- BM25 sparse retrieval not implemented; rare keyword-only queries may underperform
- Agent relies on LLM JSON output; malformed responses fall back to retrieve action
- Corpus is small (8 documents); precision metrics would improve with a larger corpus
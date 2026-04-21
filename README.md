# IDS 568 Milestone 6: RAG and Agentic Pipeline

## Model Details
- Model: mistral:7b-instruct
- Size class: 7B parameters
- Serving: Ollama (local Mac)
- Avg generation latency: 15-28s (CPU)

## Setup Instructions

1. Install Ollama from ollama.com, then run: ollama pull mistral:7b-instruct
2. Install dependencies: pip install -r requirements.txt
3. Run Part 1: Open rag_pipeline.ipynb and run all cells top to bottom
4. Run Part 2: python agent_controller.py

## Architecture Overview
Part 1: Docs > Chunker(512/64) > MiniLM > FAISS > Retriever > Mistral-7B > Answer
Part 2: Task > Agent(Mistral-7B) > [retrieve or summarize] > Final Answer

## Known Limitations
- Generation is slow on CPU (~15-28s per query)
- FAISS index is in-memory, requires re-running after restart
- BM25 sparse retrieval not implemented
- Agent uses JSON fallback when LLM output is malformed
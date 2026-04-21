"""
agent_controller.py  —  Part 2: Multi-Tool Agent
Run with: python agent_controller.py
"""

import os, json, time
import numpy as np
import faiss
import ollama
from sentence_transformers import SentenceTransformer

LLM_MODEL   = "mistral:7b-instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
os.makedirs("agent_traces", exist_ok=True)

# ── Rebuild retriever ─────────────────────────────────────────────
embedder = SentenceTransformer(EMBED_MODEL)

DOCUMENTS = [
    {"id": "doc1", "title": "RAG Overview",
     "text": "Retrieval-Augmented Generation combines a retriever and a generator. The retriever finds relevant documents from a vector store. The generator uses retrieved context to produce grounded answers. RAG reduces hallucinations because the model references real documents."},
    {"id": "doc2", "title": "Vector Databases",
     "text": "FAISS is an open-source library for efficient similarity search. Chroma adds metadata filtering and persistence. FAISS is faster for pure retrieval; Chroma is easier for filtered search."},
    {"id": "doc3", "title": "Chunking Strategies",
     "text": "Fixed-size chunking splits documents into equal-length segments with overlap. Smaller chunks give precise retrieval. Larger chunks keep more context. 512 characters with 64 overlap is a strong default."},
    {"id": "doc4", "title": "Embedding Models",
     "text": "Sentence transformers convert text into dense vectors. all-MiniLM-L6-v2 produces 384-dimensional embeddings. Cosine similarity ranks retrieved documents by relevance."},
    {"id": "doc5", "title": "Agent Controllers",
     "text": "An agent controller orchestrates multiple tools to complete multi-step tasks. Tool selection can be rule-based or LLM-driven. Transparency in decisions is critical for debugging."},
    {"id": "doc6", "title": "Evaluation Metrics",
     "text": "Precision at K measures relevant results in top-K. Recall at K measures relevant documents retrieved. Hallucination rate counts unsupported statements. End-to-end latency equals retrieval plus generation latency."},
    {"id": "doc7", "title": "LLM Serving Methods",
     "text": "Ollama is the easiest way to run open-weight LLMs locally. It supports Mistral, Llama, Qwen and others. vLLM is better for cloud GPUs. Ollama with Mistral-7B is recommended for local development."},
    {"id": "doc8", "title": "MLOps and RAG in Production",
     "text": "Production RAG needs monitoring of retrieval quality and latency. Hybrid search combines dense and BM25 sparse retrieval. Logging every step is essential for debugging production systems."},
]

all_chunks, chunk_metadata = [], []
for doc in DOCUMENTS:
    all_chunks.append(doc["text"])
    chunk_metadata.append({"doc_id": doc["id"], "title": doc["title"]})

texts = [doc["text"] for doc in DOCUMENTS]
embeddings = embedder.encode(texts, convert_to_numpy=True, batch_size=1)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"Index ready — {index.ntotal} docs")


# ── Tool 1: Retriever ─────────────────────────────────────────────
def tool_retrieve(query):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, TOP_K)
    return {
        "tool": "retriever",
        "query": query,
        "results": [
            {"title": chunk_metadata[idx]["title"],
             "text": all_chunks[idx][:300],
             "score": float(scores[0][j])}
            for j, idx in enumerate(indices[0])
        ]
    }


# ── Tool 2: Summarizer ────────────────────────────────────────────
def tool_summarize(text):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user",
                   "content": f"Summarize in 2-3 sentences:\n\n{text}"}]
    )
    return {"tool": "summarizer", "summary": response["message"]["content"]}


TOOLS = {"retrieve": tool_retrieve, "summarize": tool_summarize}


# ── Agent Decision ────────────────────────────────────────────────
def agent_decide(task, history):
    history_str = "\n".join([
        f"Step {i+1}: used '{h['tool']}'"
        for i, h in enumerate(history)
    ]) or "None yet."

    prompt = f"""You are an AI agent. Choose the next action.

Tools available:
- retrieve: search knowledge base for facts
- summarize: condense a long passage
- answer: you have enough info, write the final answer

Task: {task}
Steps so far: {history_str}

Reply in JSON only, no extra text:
{{"action": "retrieve", "input": "what to search", "reasoning": "why"}}"""

    response = ollama.chat(model=LLM_MODEL,
                           messages=[{"role": "user", "content": prompt}])
    raw = response["message"]["content"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"action": "retrieve", "input": task, "reasoning": "JSON fallback"}


def agent_final_answer(task, history):
    context = "\n\n".join([
        f"[{h['tool'].upper()}]: {json.dumps(h['result'])[:400]}"
        for h in history
    ])
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user",
                   "content": f"Using this info, answer the task.\n\nTask: {task}\n\nInfo:\n{context}\n\nAnswer:"}]
    )
    return response["message"]["content"]


# ── Main Agent Loop ───────────────────────────────────────────────
def run_agent(task, task_id, max_steps=5):
    print(f"\n{'='*55}\nTASK {task_id}: {task}\n{'='*55}")
    history = []
    trace = {"task_id": task_id, "task": task,
             "steps": [], "final_answer": "", "success": False}

    for step in range(max_steps):
        print(f"\n  [Step {step+1}] Deciding...")
        decision = agent_decide(task, history)
        action = decision.get("action", "answer")
        inp    = decision.get("input", task)
        reason = decision.get("reasoning", "")
        print(f"  → {action}: {reason[:80]}")

        step_log = {"step": step+1, "action": action,
                    "input": inp, "reasoning": reason, "result": None}

        if action == "answer":
            final = agent_final_answer(task, history)
            step_log["result"] = final
            trace["steps"].append(step_log)
            trace["final_answer"] = final
            trace["success"] = True
            print(f"\n  ANSWER: {final[:150]}...")
            break
        elif action in TOOLS:
            result = TOOLS[action](inp)
            step_log["result"] = result
            history.append({"tool": action, "result": result})
            print(f"  → Got result from {action}")
        else:
            step_log["result"] = "unknown action"

        trace["steps"].append(step_log)
    else:
        trace["final_answer"] = agent_final_answer(task, history)

    path = f"agent_traces/task_{task_id:02d}.json"
    with open(path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"  Trace saved → {path}")
    return trace


# ── 10 Evaluation Tasks ───────────────────────────────────────────
EVAL_TASKS = [
    "What is RAG and why does it reduce hallucinations?",
    "Compare FAISS and Chroma for vector storage.",
    "Explain chunking strategies and which is best for small corpora.",
    "What embedding model suits a lightweight RAG system?",
    "How does an agent controller decide which tool to use?",
    "Describe evaluation metrics for a RAG pipeline.",
    "How do I serve Mistral 7B locally using Ollama?",
    "What monitoring is needed for a production RAG system?",
    "Summarize the key components of a complete RAG pipeline.",
    "What are common failure modes in RAG-based agents?",
]

if __name__ == "__main__":
    all_traces = []
    for i, task in enumerate(EVAL_TASKS, 1):
        trace = run_agent(task, task_id=i)
        all_traces.append(trace)

    successes = sum(1 for t in all_traces if t["success"])
    print(f"\n{'='*55}")
    print(f"Done — {successes}/10 tasks succeeded")

    with open("agent_traces/summary.json", "w") as f:
        json.dump(all_traces, f, indent=2)
    print("Summary saved to agent_traces/summary.json")
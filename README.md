# RAG-Based PDF Chatbot

**Assignment:** Shridaay Techno Labs
**Candidate:** Shiva Keerth Ganti

A Retrieval-Augmented Generation (RAG) PDF chatbot that answers user questions grounded strictly in the contents of an uploaded PDF. It uses a local Ollama model to ensure privacy and accurate answers without hallucination.

## Architecture

* **Extraction:** PDF → PyMuPDF text extraction
* **Chunking:** RecursiveCharacterTextSplitter (800 chars, 150 overlap)
* **Embedding:** HuggingFace `all-MiniLM-L6-v2` (384-dim)
* **Vector Store:** ChromaDB (Cosine similarity matching)
* **Generation:** Local Ollama (`llama3.2`) with strict prompt grounding and multi-turn context

## Setup & Installation

### 1. Install Prerequisites
You will need Python 3.10+ and [Ollama](https://ollama.com/download) installed on your machine.

### 2. Pull the Local Model
Before running the application, make sure to pull the required LLM model via Ollama in your terminal:
```bash
ollama pull llama3.2
```

### 3. Install Python Dependencies
Create a virtual environment (optional but recommended) and install the pinned dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Launch the Streamlit app:
```bash
streamlit run "RAG assignment.py"
```

## Known Limitations

This project implements defensive heuristics to prioritize correctness, leading to a few known limitations:

1. **Multi-Turn Retrieval Doesn't Reformulate Queries:**
   While the LLM receives the last 4 conversation turns (to handle requests like "explain that more"), the retrieval step relies purely on embedding the *current* query. Pronouns or implicit follow-ups may not successfully fetch prior relevant chunks. A production-grade implementation would require an intermediate query reformulation step (e.g., rewriting "explain that more" to "explain [topic]").
2. **Hard Similarity Threshold (0.15):** 
   A hard similarity cutoff of `0.15` is applied to prevent the model from hallucinating when no relevant chunks are found. This threshold is specifically tuned for `all-MiniLM-L6-v2`'s typical cosine similarity range on short paraphrased queries. Changing the embedding model would require recalibrating this threshold.
3. **GPU to CPU Auto-Fallback:**
   The code attempts GPU inference first. If Ollama experiences a CUDA/GPU crash (e.g., buffer overrun), the app traps the error and forces a CPU fallback retry. This assumes the Ollama daemon successfully spawns a fresh runner process on the next request.
4. **Scanned / Image-Only PDFs:**
   The system relies on PyMuPDF text extraction. PDFs without a text layer (scans/images) will fail extraction. The app catches this gracefully and warns the user to run OCR, rather than crashing.

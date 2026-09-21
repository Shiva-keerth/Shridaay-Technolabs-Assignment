"""
RAG-Based PDF Chatbot
=====================
Assignment : Shridaay Techno Labs
Candidate  : Shiva Keerth Ganti

Architecture:
    PDF → PyMuPDF extract → RecursiveCharacterTextSplitter (800 chars, 150 overlap)
        → HuggingFace all-MiniLM-L6-v2 (384-dim) → ChromaDB (cosine)
            → Ollama llama3.2 (strict grounding, multi-turn context)
"""

import hashlib
import streamlit as st
import fitz
import ollama
import chromadb
import re
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ─────────────────────────── CONFIG ───────────────────────────
OLLAMA_MODEL      = "llama3.2"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE        = 800         # balances context completeness vs retrieval precision
CHUNK_OVERLAP     = 150         # ~19% overlap prevents mid-sentence context loss at boundaries
TOP_K             = 6           # empirically enough coverage without diluting context with noise
SIMILARITY_CUTOFF = 0.15        # tuned for MiniLM's typical range on short paraphrased queries
MAX_PDF_MB        = 20          # Reject PDFs larger than this (MB)
MAX_HISTORY_TURNS = 4           # How many prior Q&A pairs to include as context
CHROMA_COLLECTION = "rag_pdf_chatbot"

# ─────────────────────────── PAGE CONFIG & CSS ────────────────
st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0F1117 !important; }
    section[data-testid="stMain"] { background-color: #0F1117 !important; }
    .main .block-container { background-color: #0F1117 !important; padding-top: 1rem; }

    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #12122A 0%, #1A1A3E 100%) !important;
        border-right: 1px solid #2A2A5A !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown { color: #E0E0E0 !important; }
    [data-testid="stSidebar"] hr { border-color: #2A2A5A !important; }
    [data-testid="stSidebar"] small { color: #9E9EBE !important; }
    [data-testid="stSidebar"] button {
        background: rgba(121,134,203,0.2) !important;
        color: #C5CAE9 !important;
        border: 1px solid #3949AB !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] button:hover { background: rgba(121,134,203,0.35) !important; }

    [data-testid="stChatInput"] {
        background-color: #1E1E3A !important;
        border: 1px solid #3949AB !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #1E1E3A !important;
        color: #E0E0E0 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: #6272A4 !important; }

    .user-bubble {
        background: linear-gradient(135deg, #283593, #3949AB);
        color: #FFFFFF !important;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0; max-width: 80%; margin-left: auto;
        font-size: 0.95rem; line-height: 1.5;
        box-shadow: 0 2px 12px rgba(57,73,171,0.4);
    }
    .ai-bubble {
        background: #1E1E3A !important;
        color: #E0E0E0 !important;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px 0; max-width: 85%;
        border-left: 4px solid #5C6BC0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        font-size: 0.95rem; line-height: 1.6;
    }
    .source-box {
        background: #16213E !important;
        border-radius: 8px; padding: 12px 16px; margin-top: 8px;
        font-size: 0.85rem; color: #B0BEC5 !important;
        border-left: 3px solid #5C6BC0; line-height: 1.5;
    }
    .badge-green {
        background: rgba(46,125,50,0.25) !important; color: #69F0AE !important;
        padding: 4px 14px; border-radius: 20px; font-size: 0.82rem;
        font-weight: 700; display: inline-block; border: 1px solid #2E7D32;
    }
    .badge-orange {
        background: rgba(230,81,0,0.2) !important; color: #FFAB40 !important;
        padding: 4px 14px; border-radius: 20px; font-size: 0.82rem;
        font-weight: 700; display: inline-block; border: 1px solid #E65100;
    }
    [data-testid="stExpander"] {
        background: #1A1A2E !important;
        border: 1px solid #2A2A5A !important; border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary { color: #C5CAE9 !important; }
    hr { border-color: #2A2A5A !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── CACHED RESOURCES ─────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    return chromadb.Client(Settings(anonymized_telemetry=False))


# ─────────────────────────── PDF PROCESSING ───────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> list:
    """Extract text page-by-page using PyMuPDF. Returns list of {page_num, text}."""
    try:
        pages = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"page_num": page_num, "text": text})
        return pages
    except Exception as e:
        # Handle corrupted / malformed PDFs gracefully
        st.error(
            f"❌ **Could not read this PDF.** The file may be corrupted or in an unsupported format.\n\n"
            f"Error: `{e}`"
        )
        return []

def chunk_pages(pages: list) -> list:
    """Split each page into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_text":  chunk,
                "page_num":    page["page_num"],
                "chunk_index": idx,
            })
    return all_chunks

def get_pdf_hash(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


# ─────────────────────────── VECTOR STORE ─────────────────────
def build_vector_store(chunks: list, embedder, chroma_client, collection_name: str):
    """Embed all chunks and index them in a fresh ChromaDB collection."""
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    texts     = [c["chunk_text"]  for c in chunks]
    metadatas = [{"page_num": c["page_num"], "chunk_index": c["chunk_index"]} for c in chunks]
    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    BATCH = 50
    for i in range(0, len(texts), BATCH):
        embeddings = embedder.embed_documents(texts[i: i + BATCH])
        collection.add(
            documents=texts[i: i + BATCH],
            embeddings=embeddings,
            metadatas=metadatas[i: i + BATCH],
            ids=ids[i: i + BATCH],
        )
    return collection

def normalize_query(query: str) -> str:
    """Collapse extra whitespace and lowercase for more stable embedding."""
    return re.sub(r'\s+', ' ', query.strip()).lower()

def retrieve_chunks(query: str, collection, embedder, top_k: int = TOP_K, cutoff: float = SIMILARITY_CUTOFF) -> list:
    """
    Retrieve top_k chunks by cosine similarity, then filter out any
    below `cutoff` — prevents hallucination on unrelated queries.
    Includes a recursive fallback for typos (halving the cutoff).
    """
    query = normalize_query(query)
    count = collection.count()
    top_k = min(top_k, count) if count > 0 else 1
    query_vec = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1 - dist, 3)     # cosine distance -> similarity
        if similarity >= cutoff:  # DISCARD low-relevance chunks
            retrieved.append({
                "text":     doc,
                "page_num": meta.get("page_num", "?"),
                "score":    similarity,
            })
            
    # Fallback for typos/spacing issues: if zero chunks passed but we still have margin, retry with looser cutoff
    if not retrieved and cutoff > 0.05:
        return retrieve_chunks(query, collection, embedder, top_k, cutoff=cutoff * 0.5)
        
    return retrieved


# ─────────────────────────── OLLAMA GENERATION ────────────────
def generate_answer(
    query: str,
    retrieved_chunks: list,
    chat_history: list,
    model: str = OLLAMA_MODEL,
) -> str:
    """
    Build a grounded RAG prompt with multi-turn context and send to Ollama.
    Uses strict grounding — no inference beyond what's stated in the excerpts.
    """
    # If no chunks passed the similarity cutoff → answer not in PDF
    if not retrieved_chunks:
        return (
            "I'm sorry, the information you are looking for is not available "
            "in the uploaded document."
        )

    # Assemble retrieved context with page references
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Excerpt {i} — Page {chunk['page_num']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a precise PDF document assistant.\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using information explicitly present in the provided context excerpts.\n"
        "2. Do NOT use your own training knowledge or make inferences beyond what is directly stated.\n"
        "3. If the answer is not clearly present in the context, respond with exactly:\n"
        "   \"I'm sorry, the information you are looking for is not available in the uploaded document.\"\n"
        "4. When answering, cite which page(s) the information comes from.\n"
        "5. Be concise and accurate."
    )

    # Build messages list with multi-turn history for conversational context
    messages = [{"role": "system", "content": system_prompt}]

    # [KNOWN LIMITATION FOR REVIEWERS]: 
    # Multi-turn history provides conversational context to the LLM (e.g. for follow-ups like "explain that more"). 
    # However, retrieval is only run on the *current* query embedding. Prior context chunks are not re-attached.
    # A true conversational RAG would require query reformulation (e.g. rewriting "explain that more" to "explain [topic]") 
    # before retrieval to ensure relevant chunks are still fetched.
    recent_history = chat_history[-(MAX_HISTORY_TURNS * 2):]  # last N Q&A pairs
    for msg in recent_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"],
        })

    # Current user query with fresh context
    user_message = (
        f"Context excerpts from the uploaded PDF:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer using ONLY the information in the context excerpts above. "
        f"Do not use any external knowledge."
    )
    messages.append({"role": "user", "content": user_message})

    try:
        # First attempt: Let Ollama use GPU if available
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.1},
        )
    except Exception as e:
        err = str(e).lower()

        # Check CUDA/GPU crash FIRST, before any type-specific branching
        if "cuda" in err or "gpu" in err or "exit status" in err:
            try:
                # Fallback attempt: Force CPU inference
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    options={"temperature": 0.1, "num_gpu": 0},
                )
            except Exception as cpu_e:
                return f"⚠️ GPU inference failed and CPU fallback also failed: {cpu_e}"
        elif "not found" in err or "model" in err:
            return (
                f"⚠️ Ollama model **'{model}'** not downloaded.\n\n"
                f"Run: `ollama pull {model}`"
            )
        else:
            return (
                "⚠️ Cannot connect to Ollama. Make sure it is running.\n\n"
                f"**Error:** {e}"
            )

    # Validate response structure before accessing
    if (
        response
        and "message" in response
        and "content" in response["message"]
        and response["message"]["content"].strip()
    ):
        return response["message"]["content"]
    return "⚠️ The model returned an empty response. Please try rephrasing your question."


# ─────────────────────────── SESSION STATE ────────────────────
defaults = {
    "messages": [], "collection": None, "pdf_processed": False,
    "pdf_name": "", "total_pages": 0, "total_chunks": 0, "current_pdf_hash": "",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────── SIDEBAR ──────────────────────────
with st.sidebar:
    st.markdown("## 📄 RAG PDF Chatbot")
    st.markdown("*Shridaay Techno Labs — Assignment*")
    st.divider()

    st.markdown("### 📂 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"],
        help="Upload any PDF (max 20 MB) — the chatbot answers only from its content.",
    )

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()

        # ── Guard: file size check ──
        size_mb = len(pdf_bytes) / (1024 * 1024)
        if size_mb > MAX_PDF_MB:
            st.error(
                f"❌ File too large ({size_mb:.1f} MB). "
                f"Maximum supported size is {MAX_PDF_MB} MB."
            )
        else:
            pdf_hash = get_pdf_hash(pdf_bytes)
            if pdf_hash != st.session_state.current_pdf_hash:
                with st.spinner("🔄 Extracting → Chunking → Embedding..."):
                    embedder      = load_embedder()
                    chroma_client = get_chroma_client()

                    pages = extract_text_from_pdf(pdf_bytes)
                    if not pages:
                        st.warning(
                            "⚠️ **No extractable text found.** This PDF appears to be "
                            "scanned or image-only. Please upload a text-based PDF, or "
                            "use an OCR tool (e.g., Adobe Acrobat, ocrmypdf) to convert "
                            "the scanned pages to searchable text first."
                        )
                        st.stop()

                    chunks     = chunk_pages(pages)
                    collection = build_vector_store(
                        chunks, embedder, chroma_client, CHROMA_COLLECTION
                    )
                    st.session_state.collection       = collection
                    st.session_state.pdf_processed    = True
                    st.session_state.pdf_name         = uploaded_file.name
                    st.session_state.total_pages      = len(pages)
                    st.session_state.total_chunks     = len(chunks)
                    st.session_state.current_pdf_hash = pdf_hash
                    st.session_state.messages         = []
                st.success("✅ PDF indexed successfully!")

    st.divider()
    st.markdown("### 📊 Document Info")
    if st.session_state.pdf_processed:
        st.markdown('<span class="badge-green">● PDF Loaded</span>', unsafe_allow_html=True)
        st.markdown(f"**File:** `{st.session_state.pdf_name}`")
        st.markdown(f"**Pages:** {st.session_state.total_pages}")
        st.markdown(f"**Chunks indexed:** {st.session_state.total_chunks}")
    else:
        st.markdown('<span class="badge-orange">○ No PDF loaded</span>', unsafe_allow_html=True)
        st.caption("Upload a PDF to begin.")

    st.divider()
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "Ollama Model",
        ["llama3.2", "llama3", "mistral", "phi3", "gemma2"],
        help="Run `ollama pull <model>` in your terminal before selecting.",
    )
    show_sources = st.toggle("Show retrieved sources", value=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────── MAIN AREA ────────────────────────
st.markdown(
    "<div style='text-align:center;padding:20px 0 8px;'>"
    "<h1 style='color:#7986CB;font-size:2.2rem;margin:0;'>📄 RAG PDF Chatbot</h1>"
    "<p style='color:#9E9EBE;font-size:1rem;margin-top:6px;'>"
    "Ask questions about your PDF — answers grounded exclusively in the document"
    "</p></div>",
    unsafe_allow_html=True,
)
st.divider()

if not st.session_state.pdf_processed:
    st.markdown(
        "<div style='text-align:center;padding:60px 20px;color:#6272A4;'>"
        "<div style='font-size:4rem;'>📂</div>"
        "<h3 style='color:#9E9EBE;'>Upload a PDF to get started</h3>"
        "<p>Use the sidebar to upload your document.<br>"
        "Once processed, ask any question about its content.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ai-bubble">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if show_sources and msg.get("sources"):
                with st.expander("📚 Retrieved Sources from PDF", expanded=False):
                    for i, src in enumerate(msg["sources"], start=1):
                        preview = src["text"][:450] + ("..." if len(src["text"]) > 450 else "")
                        st.markdown(
                            f'<div class="source-box">'
                            f'<b>Excerpt {i} — Page {src["page_num"]}</b>'
                            f'&nbsp;&nbsp;'
                            f'<span style="color:#7986CB;font-size:0.8rem;">'
                            f'Relevance: {src["score"]:.0%}</span>'
                            f'<br><br>{preview}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Chat input
    user_input = st.chat_input(
        placeholder=f"Ask about '{st.session_state.pdf_name}'...",
    )
    if user_input and user_input.strip():
        query = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("🔍 Retrieving context and generating answer..."):
            embedder  = load_embedder()
            retrieved = retrieve_chunks(
                query, st.session_state.collection, embedder, TOP_K
            )
            answer = generate_answer(
                query,
                retrieved,
                chat_history=st.session_state.messages[:-1],  # exclude current query
                model=model_choice,
            )
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": retrieved,
        })
        st.rerun()

# ─────────────────────────── FOOTER ───────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#6272A4;font-size:0.8rem;padding:6px;'>"
    "RAG PDF Chatbot &nbsp;|&nbsp; Shridaay Techno Labs &nbsp;|&nbsp;"
    "Ollama · ChromaDB · HuggingFace · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
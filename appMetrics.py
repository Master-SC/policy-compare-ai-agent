"""
Policy & Document Comparison Assistant
AMD Hackathon — Full Version with Metrics Dashboard
"""

import re
import time
import torch
import streamlit as st
import pdfplumber
import difflib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Policy Comparison Assistant",
    page_icon="📄",
    layout="wide",
)

# ─────────────────────────────────────────
# Session state — metrics store
# ─────────────────────────────────────────
def init_metrics():
    defaults = {
        "pdf_extract_time":    None,
        "embed_time":          None,
        "llm_time":            None,
        "total_time":          None,
        "prompt_tokens":       None,
        "output_tokens":       None,
        "total_tokens":        None,
        "chunks_doc1":         None,
        "chunks_doc2":         None,
        "semantic_diffs":      None,
        "exact_diff_lines":    None,
        "avg_similarity":      None,
        "min_similarity":      None,
        "doc1_words":          None,
        "doc2_words":          None,
        "doc1_pages":          None,
        "doc2_pages":          None,
        "gpu_name":            None,
        "gpu_memory_used_gb":  None,
        "gpu_memory_total_gb": None,
        "rocm_available":      None,
        "model_name":          "meta-llama/Meta-Llama-3-8B-Instruct",
        "embed_model_name":    "all-MiniLM-L6-v2",
        "tokens_per_second":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_metrics()


# ─────────────────────────────────────────
# GPU info helper
# ─────────────────────────────────────────
def refresh_gpu_metrics():
    try:
        if torch.cuda.is_available():
            st.session_state["rocm_available"]      = True
            st.session_state["gpu_name"]            = torch.cuda.get_device_name(0)
            mem = torch.cuda.mem_get_info(0)          # (free, total) in bytes
            total_gb = mem[1] / 1024**3
            used_gb  = (mem[1] - mem[0]) / 1024**3
            st.session_state["gpu_memory_total_gb"] = round(total_gb, 1)
            st.session_state["gpu_memory_used_gb"]  = round(used_gb,  1)
        else:
            st.session_state["rocm_available"]      = False
            st.session_state["gpu_name"]            = "CPU only"
            st.session_state["gpu_memory_total_gb"] = 0
            st.session_state["gpu_memory_used_gb"]  = 0
    except Exception:
        st.session_state["rocm_available"]          = False
        st.session_state["gpu_name"]                = "Unknown"


# ─────────────────────────────────────────
# Model loading — cached once
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models (first run only)…")
def load_models():
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # AMD GPU (ROCm 7.2 + PyTorch 2.10)
    gen_pipeline = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        device=0,
    )

    # CPU fallback — uncomment if no GPU available:
    # gen_pipeline = pipeline("text-generation",
    #     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=-1)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    return embed_model, gen_pipeline, tokenizer


embed_model, summarizer, tokenizer = load_models()


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def extract_text(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text, page_count
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return "", 0


def chunk_text(text: str, min_len: int = 40) -> list[str]:
    cleaned   = re.sub(r'\n{2,}', '\n\n', text)
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    chunks    = []
    for s in sentences:
        s = s.strip().replace('\n', ' ')
        if len(s) < min_len:
            continue
        if re.match(r'^[\d\s\-–—|]+$', s):
            continue
        chunks.append(s)
    return chunks


def text_diff(a: str, b: str) -> str:
    diff = difflib.unified_diff(
        a.splitlines(), b.splitlines(),
        fromfile="Legacy Policy", tofile="Updated Policy", lineterm="",
    )
    return "\n".join(diff)


def semantic_compare(chunks1, chunks2, threshold=0.75):
    if not chunks1 or not chunks2:
        return [], 0.0, 0.0

    emb1       = embed_model.encode(chunks1, convert_to_tensor=True, show_progress_bar=False)
    emb2       = embed_model.encode(chunks2, convert_to_tensor=True, show_progress_bar=False)
    all_scores = util.cos_sim(emb1, emb2)

    results    = []
    all_best   = []
    for i, c1 in enumerate(chunks1):
        best_idx   = all_scores[i].argmax().item()
        best_score = all_scores[i][best_idx].item()
        all_best.append(best_score)
        if best_score < threshold:
            results.append((c1, chunks2[best_idx], best_score))

    results.sort(key=lambda x: x[2])
    avg_sim = sum(all_best) / len(all_best) if all_best else 0.0
    min_sim = min(all_best) if all_best else 0.0
    return results, avg_sim, min_sim


def count_tokens(text: str) -> int:
    """Count tokens using the LLM tokenizer."""
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text.split())   # rough fallback


def generate_summary(text1: str, text2: str):
    """Returns (summary_text, prompt_tokens, output_tokens, elapsed_seconds)."""
    user_prompt = f"""Analyse the two policy documents below and provide a structured summary:

1. **Key Changes** — bullet points of the most important differences
2. **Regulatory Impact** — any compliance or legal implications
3. **Risk Score** — a score from 1 (low risk) to 5 (high risk) with a one-sentence justification
4. **Reviewer Guidance** — recommended next steps for a compliance officer

---
**Document A (Legacy):**
{text1[:2500]}

---
**Document B (Updated):**
{text2[:2500]}
"""

    messages = [
        {"role": "system", "content": "You are a senior compliance analyst. Produce concise, structured policy comparison reports. Always include a numeric Risk Score (1-5)."},
        {"role": "user",   "content": user_prompt},
    ]

    prompt_tokens = count_tokens(messages[0]["content"] + messages[1]["content"])

    t0  = time.perf_counter()
    raw = summarizer(messages, max_new_tokens=500, temperature=0.3, do_sample=False)[0]["generated_text"]
    elapsed = time.perf_counter() - t0

    if isinstance(raw, list):
        summary = raw[-1].get("content", "").strip()
    else:
        pt = messages[-1]["content"]
        summary = raw[raw.index(pt) + len(pt):].strip() if pt in raw else raw.strip()

    output_tokens = count_tokens(summary)
    return summary, prompt_tokens, output_tokens, elapsed


def render_risk_score(summary_text: str):
    match = re.search(r'risk\s*score[:\s*–\-]*([1-5])', summary_text, re.IGNORECASE)
    if not match:
        return
    score  = int(match.group(1))
    colors = {1: "green", 2: "green", 3: "orange", 4: "orange", 5: "red"}
    labels = {1: "Minimal", 2: "Low", 3: "Moderate", 4: "High", 5: "Critical"}
    filled = "🟥" if score >= 4 else ("🟧" if score == 3 else "🟩")
    bar    = filled * score + "⬜" * (5 - score)
    st.markdown(f"**Risk Score:** :{colors[score]}[**{score}/5 — {labels[score]}**]  {bar}")


# ─────────────────────────────────────────
# Metrics dashboard renderer
# ─────────────────────────────────────────
def render_metrics_dashboard():
    st.subheader("📊 Run Metrics Dashboard")
    st.caption("Live performance and usage statistics for this analysis run.")

    m = st.session_state

    # ── Row 1: Token metrics ───────────────────────────────────────────────
    st.markdown("##### 🔤 Token Usage")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prompt Tokens",      f"{m['prompt_tokens']:,}"       if m['prompt_tokens']  else "—")
    c2.metric("Output Tokens",      f"{m['output_tokens']:,}"       if m['output_tokens']  else "—")
    c3.metric("Total Tokens",       f"{m['total_tokens']:,}"        if m['total_tokens']   else "—")
    c4.metric("Tokens / Second",    f"{m['tokens_per_second']:.1f}" if m['tokens_per_second'] else "—")

    st.divider()

    # ── Row 2: Latency ────────────────────────────────────────────────────
    st.markdown("##### ⏱️ Latency")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDF Extraction",    f"{m['pdf_extract_time']:.2f}s" if m['pdf_extract_time'] else "—")
    c2.metric("Embedding Time",    f"{m['embed_time']:.2f}s"       if m['embed_time']       else "—")
    c3.metric("LLM Inference",     f"{m['llm_time']:.2f}s"         if m['llm_time']         else "—")
    c4.metric("Total Run Time",    f"{m['total_time']:.2f}s"        if m['total_time']        else "—")

    st.divider()

    # ── Row 3: Document stats ─────────────────────────────────────────────
    st.markdown("##### 📄 Document Statistics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Doc A Pages",       m['doc1_pages']   if m['doc1_pages']   else "—")
    c2.metric("Doc B Pages",       m['doc2_pages']   if m['doc2_pages']   else "—")
    c3.metric("Doc A Words",       f"{m['doc1_words']:,}" if m['doc1_words'] else "—")
    c4.metric("Doc B Words",       f"{m['doc2_words']:,}" if m['doc2_words'] else "—")
    c5.metric("Chunks (Doc A)",    m['chunks_doc1']  if m['chunks_doc1']  else "—")
    c6.metric("Chunks (Doc B)",    m['chunks_doc2']  if m['chunks_doc2']  else "—")

    st.divider()

    # ── Row 4: Diff stats ─────────────────────────────────────────────────
    st.markdown("##### 🔍 Diff Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exact Diff Lines",      m['exact_diff_lines']  if m['exact_diff_lines']  is not None else "—")
    c2.metric("Semantic Changes",      m['semantic_diffs']    if m['semantic_diffs']    is not None else "—")
    c3.metric("Avg Chunk Similarity",  f"{m['avg_similarity']:.3f}" if m['avg_similarity'] else "—")
    c4.metric("Min Chunk Similarity",  f"{m['min_similarity']:.3f}" if m['min_similarity'] else "—")

    st.divider()

    # ── Row 5: GPU metrics ────────────────────────────────────────────────
    st.markdown("##### 🖥️ GPU / Hardware")
    refresh_gpu_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GPU",               m['gpu_name'] or "—")
    c2.metric("ROCm Available",    "✅ Yes" if m['rocm_available'] else "❌ No")
    c3.metric("VRAM Used",         f"{m['gpu_memory_used_gb']} GB"  if m['gpu_memory_used_gb']  else "—")
    c4.metric("VRAM Total",        f"{m['gpu_memory_total_gb']} GB" if m['gpu_memory_total_gb'] else "—")

    st.divider()

    # ── Row 6: Model info ─────────────────────────────────────────────────
    st.markdown("##### 🤖 Models")
    c1, c2 = st.columns(2)
    c1.info(f"**LLM:** `{m['model_name']}`")
    c2.info(f"**Embeddings:** `{m['embed_model_name']}`")

    # ── Export metrics as JSON ────────────────────────────────────────────
    import json
    metrics_export = {k: v for k, v in st.session_state.items()
                      if k in defaults_keys}
    st.download_button(
        "⬇️ Export metrics as JSON",
        data=json.dumps(metrics_export, indent=2, default=str),
        file_name="run_metrics.json",
        mime="application/json",
    )

defaults_keys = [
    "pdf_extract_time", "embed_time", "llm_time", "total_time",
    "prompt_tokens", "output_tokens", "total_tokens", "tokens_per_second",
    "chunks_doc1", "chunks_doc2", "semantic_diffs", "exact_diff_lines",
    "avg_similarity", "min_similarity", "doc1_words", "doc2_words",
    "doc1_pages", "doc2_pages", "gpu_name", "gpu_memory_used_gb",
    "gpu_memory_total_gb", "rocm_available", "model_name", "embed_model_name",
]


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("📄 Policy & Document Comparison Assistant")
st.caption("Upload two policy PDFs to detect exact changes, semantic shifts, and get an AI-generated executive summary.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    threshold = st.slider(
        "Semantic similarity threshold",
        min_value=0.50, max_value=0.99, value=0.75, step=0.01,
        help="Chunks below this score are flagged as changed.",
    )
    max_display = st.number_input(
        "Max semantic differences to display",
        min_value=5, max_value=100, value=20, step=5,
    )
    show_exact_diff    = st.checkbox("Show exact text diff",   value=True)
    show_semantic_diff = st.checkbox("Show semantic diff",     value=True)
    show_summary       = st.checkbox("Generate AI summary",    value=True)
    show_metrics       = st.checkbox("Show metrics dashboard", value=True)

    st.divider()
    st.caption("`all-MiniLM-L6-v2` embeddings · `Meta-Llama-3-8B-Instruct` summaries")

# ── File uploaders ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    pdf1 = st.file_uploader("📂 Legacy Policy",  type="pdf", key="legacy")
with col2:
    pdf2 = st.file_uploader("📂 Updated Policy", type="pdf", key="updated")

if not (pdf1 and pdf2):
    st.info("Upload both PDFs above to begin the comparison.")
    st.stop()

run_start = time.perf_counter()

# ── Extract text ──────────────────────────────────────────────────────────────
t0 = time.perf_counter()
with st.spinner("Extracting text from PDFs…"):
    text1, pages1 = extract_text(pdf1)
    text2, pages2 = extract_text(pdf2)
st.session_state["pdf_extract_time"] = round(time.perf_counter() - t0, 2)

# Store doc stats
st.session_state["doc1_pages"] = pages1
st.session_state["doc2_pages"] = pages2
st.session_state["doc1_words"] = len(text1.split())
st.session_state["doc2_words"] = len(text2.split())

words1, words2 = len(text1.split()), len(text2.split())
chars1, chars2 = len(text1), len(text2)

if chars1 == 0:
    st.warning("⚠️ Legacy Policy: No text extracted. The PDF may be image-only.")
    st.stop()
if chars2 == 0:
    st.warning("⚠️ Updated Policy: No text extracted. The PDF may be image-only.")
    st.stop()

c1, c2 = st.columns(2)
c1.caption(f"**Legacy Policy** · {pages1} pages · {words1:,} words · {chars1:,} chars")
c2.caption(f"**Updated Policy** · {pages2} pages · {words2:,} words · {chars2:,} chars")

st.divider()

# ── Section 1: Exact diff ─────────────────────────────────────────────────────
if show_exact_diff:
    st.subheader("🔍 Exact Text Differences")
    diff_output = text_diff(text1, text2)
    diff_lines  = len([l for l in diff_output.splitlines() if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))])
    st.session_state["exact_diff_lines"] = diff_lines

    if diff_output.strip():
        st.code(diff_output, language="diff")
        st.download_button("⬇️ Download diff as .diff", data=diff_output, file_name="policy_exact_diff.diff", mime="text/plain")
    else:
        st.success("No exact text differences found between the two documents.")
    st.divider()

# ── Section 2: Semantic diff ──────────────────────────────────────────────────
if show_semantic_diff:
    st.subheader("🧠 Semantic Differences (Meaning Changed)")
    st.caption(f"Flagging chunks with cosine similarity < {threshold:.2f}.")

    t0 = time.perf_counter()
    with st.spinner("Computing semantic embeddings…"):
        chunks1 = chunk_text(text1)
        chunks2 = chunk_text(text2)
        sem_diff, avg_sim, min_sim = semantic_compare(chunks1, chunks2, threshold=threshold)
    st.session_state["embed_time"]    = round(time.perf_counter() - t0, 2)
    st.session_state["chunks_doc1"]   = len(chunks1)
    st.session_state["chunks_doc2"]   = len(chunks2)
    st.session_state["semantic_diffs"]= len(sem_diff)
    st.session_state["avg_similarity"]= round(avg_sim, 4)
    st.session_state["min_similarity"]= round(min_sim, 4)

    total = len(sem_diff)
    shown = min(total, int(max_display))

    if total == 0:
        st.success(f"No semantic differences found at threshold {threshold:.2f}.")
    else:
        st.info(f"Found **{total}** semantically changed chunk(s). Showing top {shown}.")
        for old, new, score in sem_diff[:shown]:
            with st.expander(f"Similarity: {score:.2f} — *{old[:80]}{'…' if len(old)>80 else ''}*", expanded=False):
                c_old, c_new = st.columns(2)
                with c_old:
                    st.markdown("**Legacy text**")
                    st.markdown(f"> {old}")
                with c_new:
                    st.markdown("**Updated text**")
                    st.markdown(f"> {new}")
                st.caption(f"Cosine similarity: `{score:.4f}`")
    st.divider()

# ── Section 3: AI executive summary ──────────────────────────────────────────
if show_summary:
    st.subheader("📌 Executive Summary (AI Generated)")

    with st.status("Generating summary with Llama 3…", expanded=True) as status:
        st.write("Preparing prompt…")
        try:
            summary, prompt_tok, output_tok, llm_elapsed = generate_summary(text1, text2)
            st.session_state["prompt_tokens"]    = prompt_tok
            st.session_state["output_tokens"]    = output_tok
            st.session_state["total_tokens"]     = prompt_tok + output_tok
            st.session_state["llm_time"]         = round(llm_elapsed, 2)
            st.session_state["tokens_per_second"]= round(output_tok / llm_elapsed, 1) if llm_elapsed > 0 else 0
            status.update(label="Summary ready", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Generation failed", state="error")
            st.error(f"Model error: {e}")
            st.stop()

    render_risk_score(summary)
    st.markdown(summary)
    st.download_button("⬇️ Download summary as .md", data=summary, file_name="policy_summary.md", mime="text/markdown")
    st.divider()

# ── Total run time ────────────────────────────────────────────────────────────
st.session_state["total_time"] = round(time.perf_counter() - run_start, 2)

# ── Section 4: Metrics dashboard ─────────────────────────────────────────────
if show_metrics:
    render_metrics_dashboard()

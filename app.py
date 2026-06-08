"""
Policy & Document Comparison Assistant
"""

import re
import streamlit as st
import pdfplumber
import difflib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ─────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Policy Comparison Assistant",
    page_icon="📄",
    layout="wide",
)

# ─────────────────────────────────────────
# Model loading — cached so they load ONCE
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models (first run only)…")
def load_models():
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ── AMD GPU (ROCm / HIP) ──────────────────────────────────────────────────
    # Uses Meta-Llama-3-8B-Instruct on device 0 (AMD GPU via ROCm).
    # Requires: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
    # and HuggingFace token for gated model access.
    gen_pipeline = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        device=0,                   # 0 = first GPU (AMD via ROCm)
    )

    # ── CPU fallback (uncomment to test locally without GPU) ─────────────────
    # gen_pipeline = pipeline("text-generation", model="gpt2")

    return embed_model, gen_pipeline


embed_model, summarizer = load_models()


# ─────────────────────────────────────────
# Helper: PDF text extraction
# ─────────────────────────────────────────
def extract_text(pdf_file) -> str:
    """Extract all text from a PDF file object."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text, page_count
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return "", 0


# ─────────────────────────────────────────
# Helper: Sentence-aware chunking
# ─────────────────────────────────────────
def chunk_text(text: str, min_len: int = 40) -> list[str]:
    """
    Split text into sentence-level chunks instead of raw newlines.
    Merges very short lines and ignores boilerplate fragments.
    """
    # Collapse whitespace-only lines and join broken lines
    cleaned = re.sub(r'\n{2,}', '\n\n', text)
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    chunks = []
    for s in sentences:
        s = s.strip().replace('\n', ' ')
        # Skip fragments that look like headers, page numbers, or footers
        if len(s) < min_len:
            continue
        if re.match(r'^[\d\s\-–—|]+$', s):   # pure numbers / separators
            continue
        chunks.append(s)
    return chunks


# ─────────────────────────────────────────
# Helper: Exact text diff
# ─────────────────────────────────────────
def text_diff(a: str, b: str) -> str:
    diff = difflib.unified_diff(
        a.splitlines(),
        b.splitlines(),
        fromfile="Legacy Policy",
        tofile="Updated Policy",
        lineterm="",
    )
    return "\n".join(diff)


# ─────────────────────────────────────────
# Helper: Semantic comparison (vectorised)
# ─────────────────────────────────────────
def semantic_compare(
    chunks1: list[str],
    chunks2: list[str],
    threshold: float = 0.75,
) -> list[tuple[str, str, float]]:
    """
    Compare chunks via cosine similarity.
    Returns pairs where the best match score is below `threshold`.

    Fully vectorised — computes the entire similarity matrix in one pass
    instead of re-encoding emb1[i] inside a loop.
    """
    if not chunks1 or not chunks2:
        return []

    emb1 = embed_model.encode(chunks1, convert_to_tensor=True, show_progress_bar=False)
    emb2 = embed_model.encode(chunks2, convert_to_tensor=True, show_progress_bar=False)

    # Shape: (len(chunks1), len(chunks2))
    all_scores = util.cos_sim(emb1, emb2)

    results = []
    for i, c1 in enumerate(chunks1):
        best_idx = all_scores[i].argmax().item()
        best_score = all_scores[i][best_idx].item()
        if best_score < threshold:
            results.append((c1, chunks2[best_idx], best_score))

    # Sort by lowest similarity first (most changed)
    results.sort(key=lambda x: x[2])
    return results


# ─────────────────────────────────────────
# Helper: AI summary via Llama 3 chat template
# ─────────────────────────────────────────
def generate_summary(text1: str, text2: str) -> str:
    """
    Use the model's chat template (required for instruction-tuned Llama 3).
    Strips the prompt from the output, returning only the assistant reply.
    """
    user_prompt = f"""Analyse the two policy documents below and provide a structured summary with these sections:

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
        {
            "role": "system",
            "content": (
                "You are a senior compliance analyst. Produce concise, structured "
                "policy comparison reports. Always include a numeric Risk Score (1-5)."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    raw = summarizer(
        messages,
        max_new_tokens=500,
        temperature=0.3,
        do_sample=False,
    )[0]["generated_text"]

    # When the input is a messages list, the output is also a list of dicts.
    # Extract only the assistant's reply.
    if isinstance(raw, list):
        # Last item is the assistant turn
        return raw[-1].get("content", "").strip()

    # Fallback: strip prompt prefix from plain-text output
    prompt_text = messages[-1]["content"]
    if prompt_text in raw:
        return raw[raw.index(prompt_text) + len(prompt_text):].strip()
    return raw.strip()


# ─────────────────────────────────────────
# Helper: Parse and render risk score
# ─────────────────────────────────────────
def render_risk_score(summary_text: str):
    match = re.search(r'risk\s*score[:\s*–\-]*([1-5])', summary_text, re.IGNORECASE)
    if not match:
        return
    score = int(match.group(1))
    colors = {1: "green", 2: "green", 3: "orange", 4: "orange", 5: "red"}
    labels = {1: "Minimal", 2: "Low", 3: "Moderate", 4: "High", 5: "Critical"}
    color = colors[score]
    label = labels[score]
    filled = "🟥" if score >= 4 else ("🟧" if score == 3 else "🟩")
    empty = "⬜"
    bar = filled * score + empty * (5 - score)
    st.markdown(
        f"**Risk Score:** :{color}[**{score}/5 — {label}**]  {bar}",
        unsafe_allow_html=False,
    )


# ─────────────────────────────────────────
# Helper: PDF metadata card
# ─────────────────────────────────────────
def show_pdf_meta(label: str, text: str, pages: int):
    words = len(text.split())
    chars = len(text)
    if chars == 0:
        st.warning(f"⚠️ {label}: No text extracted. The PDF may be image-only or scanned.")
        return False
    st.caption(f"**{label}** · {pages} pages · {words:,} words · {chars:,} characters")
    return True


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("📄 Policy & Document Comparison Assistant")
st.caption("Upload two policy PDFs to detect exact changes, semantic shifts, and get an AI-generated executive summary.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    threshold = st.slider(
        "Semantic similarity threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.75,
        step=0.01,
        help="Chunks below this cosine similarity are flagged as semantically changed. Lower = more differences shown.",
    )

    max_display = st.number_input(
        "Max semantic differences to display",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Caps the number of semantic diff rows rendered to avoid browser slowdown.",
    )

    show_exact_diff = st.checkbox("Show exact text diff", value=True)
    show_semantic_diff = st.checkbox("Show semantic diff", value=True)
    show_summary = st.checkbox("Generate AI summary", value=True)

    st.divider()
    st.caption("Model: `all-MiniLM-L6-v2` for embeddings · `Meta-Llama-3-8B-Instruct` for summaries")

# ── File uploaders ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    pdf1 = st.file_uploader("📂 Legacy Policy", type="pdf", key="legacy")
with col2:
    pdf2 = st.file_uploader("📂 Updated Policy", type="pdf", key="updated")

if not (pdf1 and pdf2):
    st.info("Upload both PDFs above to begin the comparison.")
    st.stop()

# ── Extract text ──────────────────────────────────────────────────────────────
with st.spinner("Extracting text from PDFs…"):
    text1, pages1 = extract_text(pdf1)
    text2, pages2 = extract_text(pdf2)

meta_ok_1 = show_pdf_meta("Legacy Policy", text1, pages1)
meta_ok_2 = show_pdf_meta("Updated Policy", text2, pages2)

if not (meta_ok_1 and meta_ok_2):
    st.stop()

st.divider()

# ── Section 1: Exact diff ─────────────────────────────────────────────────────
if show_exact_diff:
    st.subheader("🔍 Exact Text Differences")
    diff_output = text_diff(text1, text2)
    if diff_output.strip():
        st.code(diff_output, language="diff")
        st.download_button(
            "⬇️ Download diff as .diff",
            data=diff_output,
            file_name="policy_exact_diff.diff",
            mime="text/plain",
        )
    else:
        st.success("No exact text differences found between the two documents.")

st.divider()

# ── Section 2: Semantic diff ──────────────────────────────────────────────────
if show_semantic_diff:
    st.subheader("🧠 Semantic Differences (Meaning Changed)")
    st.caption(
        f"Flagging chunks with cosine similarity < {threshold:.2f}. "
        "These passages may be paraphrased, restructured, or meaningfully altered."
    )

    with st.spinner("Computing semantic embeddings…"):
        chunks1 = chunk_text(text1)
        chunks2 = chunk_text(text2)
        sem_diff = semantic_compare(chunks1, chunks2, threshold=threshold)

    total = len(sem_diff)
    shown = min(total, int(max_display))

    if total == 0:
        st.success(f"No semantic differences found above the {threshold:.2f} threshold.")
    else:
        st.info(
            f"Found **{total}** semantically changed chunk(s). "
            f"Displaying the {shown} most changed."
        )

        for old, new, score in sem_diff[:shown]:
            with st.expander(
                f"Similarity: {score:.2f} — *{old[:80]}{'…' if len(old) > 80 else ''}*",
                expanded=False,
            ):
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
            summary = generate_summary(text1, text2)
            status.update(label="Summary ready", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Generation failed", state="error")
            st.error(f"Model error: {e}")
            st.stop()

    render_risk_score(summary)
    st.markdown(summary)

    st.download_button(
        "⬇️ Download summary as .md",
        data=summary,
        file_name="policy_summary.md",
        mime="text/markdown",
    )
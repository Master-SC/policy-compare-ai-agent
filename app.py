import streamlit as st
import pdfplumber
import difflib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# -----------------------------
# Load Models
# -----------------------------

# Embedding model for semantic comparison
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Summarizer (CPU-friendly)
summarizer = pipeline(
    "text-generation",
    model="gpt2"
)

# -----------------------------
# Helper Functions
# -----------------------------

def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text):
    return [p.strip() for p in text.split("\n") if len(p.strip()) > 20]


def semantic_compare(chunks1, chunks2):
    emb1 = model.encode(chunks1, convert_to_tensor=True)
    emb2 = model.encode(chunks2, convert_to_tensor=True)

    results = []
    for i, c1 in enumerate(chunks1):
        sim_scores = util.cos_sim(emb1[i], emb2)[0]
        best_match_idx = sim_scores.argmax().item()
        best_score = sim_scores[best_match_idx].item()

        if best_score < 0.75:
            results.append((c1, chunks2[best_match_idx], best_score))

    return results


def text_diff(a, b):
    diff = difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="")
    return "\n".join(diff)


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("📄 Policy & Document Comparison Assistant")

pdf1 = st.file_uploader("Upload Legacy Policy", type="pdf")
pdf2 = st.file_uploader("Upload Updated Policy", type="pdf")

if pdf1 and pdf2:

    # Extract text
    text1 = extract_text(pdf1)
    text2 = extract_text(pdf2)

    # -----------------------------
    # Exact Diff
    # -----------------------------
    st.subheader("🔍 Exact Text Differences")
    st.code(text_diff(text1, text2))

    # -----------------------------
    # Semantic Diff
    # -----------------------------
    st.subheader("🧠 Semantic Differences (Meaning Changed)")
    chunks1 = chunk_text(text1)
    chunks2 = chunk_text(text2)
    sem_diff = semantic_compare(chunks1, chunks2)

    for old, new, score in sem_diff:
        st.markdown(f"**Old:** {old}")
        st.markdown(f"**New:** {new}")
        st.markdown(f"Similarity Score: {score:.2f}")
        st.markdown("---")

    # -----------------------------
    # AI Summary
    # -----------------------------
    st.subheader("📌 Executive Summary (AI Generated)")

    summary_input = f"""
    Summarize the key differences between the two documents below.
    Highlight:
    - Key changes
    - Regulatory impact
    - Risk score (1 to 5)
    - Reviewer guidance

    Document A:
    {text1[:2000]}

    Document B:
    {text2[:2000]}
    """

    with st.spinner("Generating summary..."):
        raw_output = summarizer(
            summary_input,
            max_length=300,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )[0]["generated_text"]

    st.write(raw_output)

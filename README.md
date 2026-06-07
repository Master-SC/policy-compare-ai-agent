📄 Policy & Document Comparison Assistant
AI‑powered tool for comparing policy documents with text diff, semantic diff, and LLM‑generated executive summaries.

🚀 Overview
The Policy & Document Comparison Assistant helps organizations quickly understand changes between two versions of a policy or regulatory document. It highlights:

🔍 Exact text differences (additions, deletions, modifications)

🧠 Semantic differences (meaning‑level changes using embeddings)

📌 AI‑generated executive summary (key changes, regulatory impact, risk score)

⚡ Fast, clean Streamlit UI


✨ Features
✔ Exact Text Diff
Color‑coded unified diff showing additions and deletions.

✔ Semantic Diff
Uses SentenceTransformer embeddings to detect meaning‑level changes even when wording is different.

✔ AI Executive Summary
Generates a concise summary including:

Key changes

Regulatory impact

Risk score (1–5)

Reviewer guidance

✔ Streamlit UI
Simple, clean interface for uploading and comparing PDFs.

✔ AMD Cloud Ready
Local summarizer uses GPT‑2 for compatibility.
On AMD Cloud, the summarizer switches to Llama‑3 for high‑quality GPU‑accelerated inference.

🛠️ Tech Stack
Component	Technology
UI	Streamlit
Text Extraction	pdfplumber
Semantic Embeddings	SentenceTransformer (MiniLM‑L6‑v2)
Local Summarizer	GPT‑2 (text-generation)
AMD Cloud Summarizer	Llama‑3‑8B‑Instruct
Diff Engine	Python difflib


📦 Installation
1. Clone the repository
bash
git clone https://github.com/<your-username>/policy-compare-assistant.git
cd policy-compare-assistant
2. Create a virtual environment
bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
3. Install dependencies
bash
pip install -r requirements.txt
▶️ Running the App
bash
streamlit run app.py
Then open the browser at:

Code
http://localhost:8501
Upload two PDF policy documents and view:

Exact diff

Semantic diff

AI summary

⚡ AMD Cloud Deployment (Llama‑3)
On AMD Cloud:

Install ROCm PyTorch

Install latest transformers

Replace local summarizer with:

python
summarizer = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    device=0
)
This enables:

GPU‑accelerated summarization

Faster inference

Higher quality summaries

GPU metrics for hackathon PPT

📁 Project Structure
Code

policy-compare-assistant/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
├── sample_pdfs/          # Example PDFs (ignored in git)
├── .gitignore            # Clean ignore rules
└── .venv/                # Virtual environment (ignored)


🧪 Sample Use Cases
Policy updates

Compliance audits

HR document revisions

Regulatory change tracking

Legal document comparison

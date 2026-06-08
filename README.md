# 📄 Policy & Document Comparison Assistant

> An AI-powered document comparison tool built for running on AMD Instinct MI300X GPUs via ROCm. Upload two policy PDFs and instantly get exact diffs, semantic change detection, and an AI-generated executive summary.

---

## 🚀 Features

- **Exact Text Diff** — Line-by-line comparison using `difflib`, rendered as a colour-coded unified diff
- **Semantic Diff** — Detects paraphrased or restructured clauses using `sentence-transformers` (MiniLM) and cosine similarity
- **AI Executive Summary** — Powered by Meta Llama 3 8B Instruct; highlights key changes, regulatory impact, risk score (1–5), and reviewer guidance
- **Risk Score Visualiser** — Parses and renders a colour-coded risk indicator from the AI summary
- **Configurable Threshold** — Sidebar slider to tune semantic sensitivity on the fly
- **Download Reports** — Export the exact diff (`.diff`) and AI summary (`.md`) as files
- **AMD GPU Optimised** — Uses ROCm 7.0 + PyTorch 2.6 for GPU-accelerated inference on MI300X

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI                        │
├──────────────┬──────────────────┬───────────────────┤
│  Exact Diff  │  Semantic Diff   │   AI Summary       │
│  (difflib)   │  (MiniLM + cos)  │  (Llama 3 8B)     │
├──────────────┴──────────────────┴───────────────────┤
│              PDF Text Extraction (pdfplumber)        │
├─────────────────────────────────────────────────────┤
│         AMD Instinct MI300X · ROCm 7.0 · PyTorch 2.6│
└─────────────────────────────────────────────────────┘
```

---

## 🖥️ Running on AMD Developer Cloud (MI300X)

### 1. Create a GPU Droplet

1. Go to [cloud.amd.com](https://cloud.amd.com)
2. Select **Create GPU Droplet**
3. Choose region: **ATL1** (Atlanta — only region with AMD GPUs)
4. Select image: **PyTorch 2.6.0 · ROCm 7.0.0**
5. Select plan: **MI300X** (1 GPU · 192 GB VRAM · $1.99/hr)
6. Add your SSH key and launch

### 2. Connect to the instance

```bash
ssh root@<your-droplet-ip>
```

### 3. Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  AMD Instinct MI300X
```

### 4. Install dependencies

```bash
pip install streamlit pdfplumber sentence-transformers transformers accelerate huggingface_hub
```

### 5. Authenticate with HuggingFace (required for Llama 3)

Llama 3 is a gated model. Accept Meta's licence at [huggingface.co/meta-llama](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) then:

```bash
huggingface-cli login
# Paste your HF token when prompted
```

### 6. Upload and run the app

```bash
# From your local machine
scp policy_comparison_app.py root@<your-droplet-ip>:~/

# On the instance
streamlit run policy_comparison_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0
```

Open `http://<your-droplet-ip>:8501` in your browser.

> ⚠️ **Important:** Destroy the droplet when done — powered-off droplets still incur charges as resources stay reserved.

---

## 💻 Running Locally (CPU Mode)

No GPU? No problem. The full pipeline works on any laptop with one model swap.

### 1. Install dependencies

```bash
pip install streamlit pdfplumber sentence-transformers transformers accelerate
```

### 2. Swap the model for a CPU-friendly alternative

In `policy_comparison_app.py`, inside `load_models()`, replace:

```python
# AMD GPU version
gen_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    device=0
)
```

with:

```python
# CPU version
gen_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1  # -1 = CPU
)
```

### 3. Run

```bash
streamlit run policy_comparison_app.py
```

Open `http://localhost:8501` in your browser.

### Local performance guide

| RAM   | Experience                                      |
|-------|-------------------------------------------------|
| 8 GB  | Works — semantic diff may be slow on large PDFs |
| 16 GB | ✅ Smooth for most documents                    |
| 32 GB | ✅ Fast — can try larger models                 |

---

## 📦 Requirements

```txt
streamlit
pdfplumber
sentence-transformers
transformers
accelerate
huggingface_hub
torch  # with ROCm 7.0 on AMD GPU, or standard CPU build locally
```

Install all at once:

```bash
pip install streamlit pdfplumber sentence-transformers transformers accelerate huggingface_hub
```

For AMD GPU (ROCm):

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.0
```

---

## ⚙️ Configuration

All settings are available in the **sidebar** at runtime — no code changes needed:

| Setting | Default | Description |
|---|---|---|
| Semantic similarity threshold | `0.75` | Chunks below this score are flagged as changed. Lower = more differences shown |
| Max semantic differences | `20` | Caps rendered rows to prevent browser slowdown |
| Show exact diff | `on` | Toggle the unified diff section |
| Show semantic diff | `on` | Toggle the semantic comparison section |
| Generate AI summary | `on` | Toggle the Llama 3 summary section |

---

## 🗂️ Project Structure

```
.
├── app.py                     # Main Streamlit application
├── requirments.txt            # Dependency List
├── .gitignore                 # gitignore file
└── README.md                  # This file
```

---

## 🔑 Key Technical Decisions

**Vectorised semantic comparison** — The full cosine similarity matrix is computed in a single `util.cos_sim(emb1, emb2)` call instead of looping per chunk. This is O(1) GPU operations regardless of document size.

**Model caching** — Both the embedding model and LLM pipeline are wrapped in `@st.cache_resource`, ensuring they load exactly once per session rather than on every Streamlit interaction.

**Chat template for Llama 3** — The summariser receives a structured `[system, user]` messages list to invoke the instruction-tuned model correctly. The assistant reply is extracted from the final dict, preventing prompt bleed in the output.

**Sentence-aware chunking** — Text is split on sentence-ending punctuation rather than raw newlines, filtering out page numbers and short fragments that would pollute semantic diff results.

---

## 🏆 Built For

 showcasing AMD Instinct MI300X GPU acceleration for real-world AI document workflows.

- GPU: AMD Instinct MI300X (192 GB HBM3)
- Software stack: ROCm 7.0 · PyTorch 2.6 · HuggingFace Transformers
- Cloud: AMD Developer Cloud (powered by DigitalOcean) · ATL1

---

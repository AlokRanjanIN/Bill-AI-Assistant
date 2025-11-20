# Bill Assistant

This repository contains a Streamlit app and a `BillAssistant` class for extracting structured data from bills/invoices (images or PDFs) using an OCR model and semantic search. The app provides a simple web UI to upload a bill, run OCR + parsing, inspect extracted text, download JSON, and ask chat-style questions about the bill.

> **Note:** The repo includes optional integration with the Groq LLM client for LLM-powered parsing and answers. A `GROQ_API_KEY` is **optional** but required to enable the Groq-based parsing/LLM flows.

---

## Repo structure

```
main.py             # Streamlit app (web UI)
assistant.py        # BillAssistant class and logic (OCR, parsing, embeddings, search)
requirements.txt    # Python dependencies (see notes)
BillAssistant.ipynb  # Colab notebook variant
```

---

## Prerequisites

* Python 3.8+ (3.9/3.10 recommended)

**Important note for the DeepSeek OCR model:** If you plan to use the `deepseek-ai/DeepSeek-OCR` model (the default `model_name` in `assistant.py`), the model has stricter runtime requirements: it requires **Python 3.12 or newer**, **does not support CPU-only inference**, and **must be run on a single GPU (dual-GPU / multi-GPU setups are not supported by the provided model configuration)**. If your environment does not meet these constraints, either use a different OCR checkpoint that supports your platform or run in an environment (e.g., a single-GPU VM) that satisfies these requirements.

* `pip` installed
* (Optional, recommended) A machine with GPU + CUDA if you plan to load large models locally. The code tries to auto-detect and use CUDA when available.

Important libraries used that may need manual installs or special wheels:

* `torch` (install the correct version for your CUDA)
* `transformers`, `sentence-transformers`
* `PyMuPDF` (for PDF → image conversion)
* `streamlit`
* `rank-bm25`, `nltk`
* `flash-attn` (optional; can be difficult to install on some platforms)

> **Tip:** If you run into `flash-attn` installation issues, remove or comment it out in `requirements.txt` and/or `assistant.py` and run without it. The system works fine without it.

---

## Quick start (local)

1. **Clone the repo**

```bash
git clone https://github.com/AlokRanjanIN/Bill-AI-Assistant.git
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies**

```
pip install -r requirements.txt
```

If you do not already have `torch` installed (strongly recommended), install it separately following the instructions for your CUDA or CPU setup from the official PyTorch site. For example (CPU-only):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Also install `sentence-transformers` which is required by `assistant.py`:

```bash
pip install sentence-transformers
```

4. **(Optional) Provide Groq API key**

* If you plan to enable Groq-based LLM parsing and responses, set a `GROQ_API_KEY`.

Two options:

**A) Run in Google Colab**

The code currently supports fetching `GROQ_API_KEY` when running in Colab via `google.colab.userdata`. If you use the included Colab notebook, store the key there.

**B) Run locally**

Set an environment variable before starting Streamlit (Linux / macOS):

```bash
export GROQ_API_KEY="your_groq_api_key_here"
streamlit run main.py
```

On Windows (PowerShell):

```powershell
$env:GROQ_API_KEY = "your_groq_api_key_here"
streamlit run main.py
```

> NOTE: The `BillAssistant` object in `main.py` is constructed with `use_colab_secrets=True`. This tries to load the key using Colab utilities (intended for Colab runs). To use an environment variable locally, either:
>
> 1. Change `main.py` to call `BillAssistant(use_colab_secrets=False)` and modify `assistant.py` to read `os.environ.get('GROQ_API_KEY')`, or
> 2. Export `GROQ_API_KEY` to the environment and adapt `assistant.py` accordingly (see snippet below).

**Sample change to `assistant.py` (local GROQ support)**

```python
# inside __init__ of BillAssistant, after colab logic
if self.client is None:
    # try local env var
    groq_key = os.environ.get('GROQ_API_KEY')
    if groq_key:
        try:
            self.client = Groq(api_key=groq_key)
            print("✅ Groq client initialized from environment variable.")
        except Exception as e:
            print(f"⚠️ Failed to init Groq client from env: {e}")
```

5. **Run the Streamlit app**

```bash
streamlit run main.py
```

This will open a local Streamlit web UI (usually at `http://localhost:8501`). Upload an image or PDF bill and click **Process bill**.

---

## What the app does

* Accepts image (PNG/JPEG) or PDF uploads.
* Runs OCR via the `BillAssistant.model_run` (calls the `model.infer()` interface when a model is available; otherwise falls back to no-extraction behavior).
* Parses the extracted text into structured JSON using the Groq LLM client (if available) or returns a minimal fallback.
* Builds sentence-transformer embeddings and a small BM25 index for semantic Q&A and retrieval over bill text.
* Shows a summary (invoice number, date, total), displays line items in a table, and lets you download a JSON representation.
* Provides a chat UI to ask questions about the processed bill.

---

## Architecture & data flow (high level)

This section explains how data flows through the system and the role of each component.

### Components

* **Streamlit UI (`main.py`)** — User-facing web interface. Handles file uploads, preview, UI buttons (Process, Clear, Show raw text), displays results and chat messages, and caches the `BillAssistant` instance so heavy models load only once per session.

* **BillAssistant (`assistant.py`)** — Core logic and orchestration. Responsible for:

  * Running OCR (`model_run`) using the configured OCR model (e.g., `deepseek-ai/DeepSeek-OCR`),
  * Parsing extracted text into structured JSON (`parse_bill_with_llm`) using Groq LLM (optional),
  * Chunking text and computing sentence-transformer embeddings for semantic search,
  * Building a BM25 index for keyword search (hybrid retrieval),
  * Answering chat queries either via rule-based lookups, hybrid semantic search, or by calling the Groq LLM to formulate precise answers.

* **OCR Model (transformers checkpoint)** — The heavy model that performs visual understanding and OCR on images. May require specific Python and hardware constraints (see prerequisites). When not available locally, `model_run` will fall back and return an empty string.

* **Sentence-transformer (`all-MiniLM-L6-v2`)** — Lightweight model used to compute embeddings for chunks of bill text to support semantic similarity queries.

* **Groq LLM client (optional)** — Used for robust JSON parsing of OCR output and for LLM-driven answers. Disabled if no `GROQ_API_KEY` or client initialization fails.

* **PyMuPDF (`fitz`)** — Used to convert PDF pages into images before running OCR.

* **BM25 (rank-bm25)** — Classical keyword-based ranking used in hybrid search.

### Flow (step-by-step)

1. **Upload:** User uploads an image or a PDF in the Streamlit UI.
2. **Persist:** `main.py` saves uploaded content to a temporary file to make it accessible to models and processing code.
3. **OCR (model_run):** For images (or PDF pages converted to images), `BillAssistant.model_run` invokes the OCR model to extract raw text. If the model is missing/unavailable, a fallback behavior is used (usually empty text or a debug string).
4. **Parsing (LLM):** The raw OCR text is passed to `parse_bill_with_llm`. If Groq is available, the assistant asks the LLM to return a strict JSON object with invoice fields. The assistant attempts to robustly repair and `json.loads()` the LLM output with `safe_load_json_recover`.
5. **Store structured bill:** Parsed JSON is stored in `self.current_bill`, and the raw extracted text is stored in `self.bill_text` for inspection and downstream processing.
6. **Chunking & embeddings:** The raw text is split into overlapping chunks; sentence-transformer embeddings are computed for each chunk and a whole-bill embedding is also calculated.
7. **BM25 index:** A BM25 index is created over preprocessed chunk tokens (stopwords removed) to support keyword retrieval.
8. **Hybrid search availability:** When answering queries, the assistant computes semantic similarity scores (cosine) and BM25 scores, normalizes both, and combines them with an `alpha` weight to produce a hybrid ranking.
9. **Answering / LLM fallback:** For simple questions (total, date, invoice number, items), the assistant can answer via direct dictionary lookup. For more complex questions it uses hybrid retrieval to gather relevant chunks and — if Groq is available — asks the LLM to produce a concise answer using only the retrieved context.
10. **UI output & export:** Streamlit displays the parsed fields, a table of line items (if available), allows the user to download JSON, and provides a chat interface.

### Sequence diagram (textual)

User → Streamlit UI (`main.py`): upload file → Streamlit → saves file → BillAssistant.process_bill(image_path)
BillAssistant → (if PDF) PyMuPDF: convert pages → for each page call model_run → OCR model → returns text
BillAssistant → parse_bill_with_llm (Groq LLM if available) → returns structured JSON
BillAssistant → chunk_text + sentence-transformer → compute embeddings → build BM25
User ↔ Streamlit chat → BillAssistant.chat/answer_query → (hybrid search ± Groq) → results displayed

---

## Design decisions & rationale

* **Hybrid search (semantic + BM25):** Combines the strengths of dense embeddings (semantic similarity) and BM25 (keyword precision) — helpful when OCR output is noisy or when short, literal tokens like invoice numbers are important.

* **Chunking with overlap:** Overlap helps preserve context that might otherwise be split between chunks and improves retrieval quality.

* **Robust JSON recovery:** LLM outputs can be malformed or truncated; `safe_load_json_recover` attempts multiple heuristics (cleaning, minimal repairs, balancing braces, progressive trimming) before failing to speed up iteration and debugging.

* **Streamlit caching:** `@st.cache_resource` keeps the heavy `BillAssistant` instance in memory across user interactions to avoid reloading big models repeatedly.

* **Optional Groq LLM:** Keeping the LLM client optional lets the repo be used locally without an API key; Groq provides a convenient way to produce structured JSON from messy OCR text when available.

---

## Where to extend / improve

* **Stronger OCR fallback:** Add an open-source OCR path (Tesseract, PaddleOCR) when the DeepSeek checkpoint is unavailable.
* **Configurable chunking/tokenization:** Use tokenizers (e.g., `tiktoken`) so chunk sizes match model token limits precisely.
* **Persistent storage:** Add a database (SQLite/Postgres) to store processed bills and audit logs rather than keeping everything in memory.
* **Authentication & secrets management:** Use Streamlit Secrets or `.env` and avoid printing API keys. Add an example `secrets.toml` for Streamlit Cloud deploys.
* **Tests & CI:** Add unit tests for JSON repair, chunking, BM25 scoring, and simple end-to-end tests with small sample bills.

---

## Notes, tips & troubleshooting

* **Model & memory**: Loading large models locally will use large amounts of RAM / VRAM. If you get `CUDA out of memory` errors, try reducing device_map usage or run on CPU.
* **OCR/model not available**: If `self.model` or tokenizer fails to load (for example, because the checkpoint is not available), `model_run` will print a warning and return an empty string. In this case parsing will fail — use a supported OCR model or run the Colab notebook where the model files may be present.
* **Flash-attn**: optional optimization; if build fails during `pip install`, remove it from `requirements.txt` and re-install.
* **PDFs**: The app uses PyMuPDF (`fitz`) to convert PDF pages to PNGs. If PDF processing fails, ensure `PyMuPDF` is installed properly.
* **NLTK stopwords**: The code attempts to download `stopwords` at runtime. If your environment blocks downloads, pre-download them manually:

```python
python -c "import nltk; nltk.download('stopwords')"
```

* **Permissions / temp files**: Uploaded files are saved to a temporary file. If you change permissions or run in a restricted environment, adjust `save_uploaded_to_temp` in `main.py` accordingly.

* **Session caching**: Streamlit caches the `BillAssistant` instance with `@st.cache_resource` so models only load once per session. If you change code and want to reload resources, use the `Clear / Reset` button in the UI or restart Streamlit.

---

## Security & privacy

* Uploaded files may be saved temporarily on disk. Do not use this application to process sensitive documents unless you understand how/where those files are stored and cleaned up.
* If you enable the Groq LLM, bill text will be sent to an external API — ensure this is acceptable for your data privacy requirements.

---

## Development / modifying behavior

* To force a local-only mode (no Groq): in `main.py` change `BillAssistant(use_colab_secrets=True)` to `BillAssistant(use_colab_secrets=False)`. Then the assistant will skip Colab-specific secret loading.
* To change the parsing prompt or model parameters, edit `assistant.py::process_bill` and `assistant.py::parse_bill_with_llm`.

---

## Example: run without Groq (local only)

1. Edit `main.py` line that constructs `BillAssistant`:

```python
# before
assistant = load_assistant()
# where load_assistant returns BillAssistant(use_colab_secrets=True)

# change to
@st.cache_resource
def load_assistant():
    return BillAssistant(use_colab_secrets=False)
```

2. Restart Streamlit and process a bill. You will still get OCR output (if model is available) and local semantic search; the LLM-based JSON parsing step will be disabled unless you implement a local LLM client.

---

## License

This repository is provided as-is for demonstration / research / development purposes. No warranty. Adapt as needed.

---

## Want help?

If you want, I can:

* Add a sample `secrets.toml` or `.env` example for Streamlit.
* Create a smaller `requirements-lite.txt` that omits heavy packages like `flash-attn`.
* Patch `assistant.py` to prefer environment `GROQ_API_KEY` when not running in Colab (I included a snippet above).

Just tell me which you'd like and I will update the README or repository accordingly.

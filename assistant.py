import os
import json
import warnings
from datetime import datetime
from io import BytesIO
import tempfile  # For PDF processing

# Third-party libs
import torch
import gc

# Transformers / sentence-transformers
from transformers import AutoModel, AutoTokenizer, logging as hf_logging
from sentence_transformers import SentenceTransformer

# Aux libs used in CLI
import requests
from PIL import Image
import pandas as pd
import numpy as np
import re
from tabulate import tabulate

import flash_attn  # optional

# Basic environment / warning control
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", message=".*GetPrototype.*")
hf_logging.set_verbosity_error()

from google.colab import userdata
colab_secrets_available = True
from groq import Groq
import fitz  # PyMuPDF
PDF_SUPPORT = True

from rank_bm25 import BM25Okapi
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
HYBRID_SEARCH_AVAILABLE = True

class BillAssistant:
    """
    Class-based bill assistant with semantic Q&A using sentence-transformer embeddings.

    Usage:
        assistant = BillAssistant(model_name='deepseek-ai/DeepSeek-OCR')
        assistant.run_cli()
    """

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR", use_colab_secrets: bool = False):
        self.model_name = model_name
        self.device_info = self._gather_device_info()
        self.model = None
        self.tokenizer = None
        self.sentence_model = None
        self.client = None  # Groq client (optional)
        self.current_bill = None
        self.bill_text = None
        self.pdf_support = PDF_SUPPORT

        # Semantic structures
        self.chunks = []               # list[str]
        self.chunk_embeddings = None   # numpy.ndarray shape (n_chunks, emb_dim)
        self.bill_embeddings = None    # embedding of whole bill (optional)

        # Hybrid search components
        self.bm25 = None
        self.bm25_corpus = None
        self.hybrid_search_available = HYBRID_SEARCH_AVAILABLE

        # Optionally load Colab secrets and Groq client
        if use_colab_secrets and colab_secrets_available and Groq is not None:
            try:
                GROQ_API_KEY = userdata.get("GROQ_API_KEY")
                if GROQ_API_KEY:
                    self.client = Groq(api_key=GROQ_API_KEY)
            except Exception as e:
                print(f"⚠️ Failed to init Colab secrets / Groq client: {e}")

        # Load models eagerly (you may want to lazy-load in heavy environments)
        self.load_models()
        self.print_device_info()

    # ------ Utility / device info ------
    def _gather_device_info(self):
        try:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                try:
                    device_name = torch.cuda.get_device_name(0)
                except Exception:
                    device_name = "Unknown CUDA device"
                try:
                    compute_cap = torch.cuda.get_device_capability(0)
                except Exception:
                    compute_cap = ("N/A",)
            else:
                device_name = "CPU"
                compute_cap = ("N/A",)
            return {
                "torch_version": torch.__version__,
                "cuda_available": cuda_available,
                "device_name": device_name,
                "compute_capability": compute_cap,
                "flash_attn": getattr(flash_attn, "__version__", None) if flash_attn else None
            }
        except Exception as e:
            return {"error": str(e)}

    def print_device_info(self):
        info = self.device_info
        print(f"PyTorch version: {info.get('torch_version')}")
        print(f"CUDA available: {info.get('cuda_available')}")
        print(f"GPU: {info.get('device_name')}")
        print(f"Compute capability: {info.get('compute_capability')}")
        if info.get("flash_attn"):
            print(f"✓ Flash Attention version: {info.get('flash_attn')}")
        else:
            print("✗ Flash Attention not installed or not available")

    # ------ Model loading ------
    def load_models(self):
        "Load tokenizer, model and sentence-transformer used for embeddings."
        print("Loading Assistant Eye...", end='\t')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                use_safetensors=True
            )
            self.model = self.model.eval()
            print("✅ ASSISTANT EYE LOADED SUCCESSFULLY!!!")
        except Exception as e:
            print(f"⚠️ Failed to load model/tokenizer: {e}")
            self.model = None
            self.tokenizer = None

        try:
            print("Loading Assistant Brain...", end='\t')
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ ASSISTANT BRAIN LOADED SUCCESSFULLY!!!")
        except Exception as e:
            print(f"⚠️ Failed to load sentence-transformer: {e}")
            self.sentence_model = None

    # ------ Core OCR / inference run ------
    def model_run(self, prompt: str, image_file: str):
        """
        Run the OCR/inference model.
        - If the real model is available, call model.infer(...) as per original script.
        - If not available (or for debugging), returns a hardcoded sample result.
        """
        output_path = f"/content/outputs/{os.path.splitext(os.path.basename(image_file))[0]}"
        os.makedirs(output_path, exist_ok=True)

        if self.model is None or self.tokenizer is None:
            # fallback -- return debug sample text (same as your test)
            print("⚠️ Model/tokenizer unavailable.")
            return ""

        # If real model exists, call its inference method (kept as in original code)
        try:
            print("OCR...")
            torch.cuda.empty_cache()
            res = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_file,
                output_path=output_path,
                base_size=1536,
                image_size=1024,
                crop_mode=False,
                save_results=True,
                test_compress=False,
                eval_mode=True,  # return instead of printing
            )
            print(f"Extraction complete.\n{res}")
            return res
        except Exception as e:
            print(f"⚠️ Model inference failed: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50):
        """
        Split text into overlapping chunks (approx. chunk_size tokens/characters).
        This uses naive character-based splitting for simplicity. For production,
        use token-based splitting (e.g., tiktoken) to respect model token counts.
        """
        if not text:
            return []
        text = text.strip()
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap  # overlap
        return chunks

    def compute_chunk_embeddings(self):
        """
        Compute embeddings for each chunk and also store embedding for whole bill.
        """
        if self.sentence_model is None:
            print("⚠️ Sentence model not available; cannot compute embeddings.")
            self.chunk_embeddings = None
            self.bill_embeddings = None
            self.bm25 = None
            self.bm25_corpus = None
            return

        if not self.chunks:
            self.chunk_embeddings = None
            self.bill_embeddings = None
            self.bm25 = None
            self.bm25_corpus = None
            return

        print("🧠 Computing semantic embeddings...")
        emb = self.sentence_model.encode(self.chunks, convert_to_numpy=True)
        # Normalize embeddings (helps cosine similarity)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        emb_norm = emb / norms
        self.chunk_embeddings = emb_norm  # shape (n_chunks, d)

        # whole-bill embedding
        whole_emb = self.sentence_model.encode([self.bill_text], convert_to_numpy=True)
        whole_emb /= (np.linalg.norm(whole_emb, axis=1, keepdims=True) + 1e-10)
        self.bill_embeddings = whole_emb[0]

       # Initialize BM25 for hybrid search if available
        if self.hybrid_search_available:
            print("📚 Building BM25 index for hybrid search...")
            try:
                self.bm25_corpus = [self._preprocess_text(chunk) for chunk in self.chunks]
                self.bm25 = BM25Okapi(self.bm25_corpus)
                print(f"✅ BM25 index built with {len(self.chunks)} chunks!")
            except Exception as e:
                print(f"⚠️ Failed to build BM25 index: {e}")
                self.bm25 = None
                self.bm25_corpus = None
        else:
            self.bm25 = None
            self.bm25_corpus = None

    # ------ High-level processing ------
    def process_bill(self, image_path: str, prompt: str = "<image>\nStrict OCR. Extract all the text in the image as Markdown."):
        """
        Process a bill image or PDF: run OCR, parse to structured JSON, and compute embeddings.
        Returns a status message (string).
        """
        if not image_path:
            return "❌ No image path provided."

        # If URL, download
        if image_path.startswith("http"):
            try:
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                tmp_path = "/content/tmp/bill_download"
                _, ext = os.path.splitext(image_path)
                tmp_path += ext.lower() if ext else ".jpg"
                img.save(tmp_path)
                image_path = tmp_path
            except Exception as e:
                return f"❌ Failed to download image: {e}"

        if not os.path.exists(image_path):
            return "❌ File not found!"

        # Handle PDF files
        if image_path.lower().endswith('.pdf'):
            if not self.pdf_support:
                return "❌ PDF processing not available. Install PyMuPDF with: pip install PyMuPDF"

            print("📄 Processing PDF file (converting pages to images)...")
            temp_dir = tempfile.mkdtemp()
            image_paths = []
            try:
                # Convert PDF to images
                doc = fitz.open(image_path)
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    pix = page.get_pixmap(dpi=150)  # 150 DPI for good quality
                    img_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
                    pix.save(img_path)
                    image_paths.append(img_path)
                doc.close()

                # Process each page
                bill_texts = []
                for i, img_path in enumerate(image_paths):
                    print(f"📝 Extracting text from page {i+1}/{len(image_paths)}...")
                    page_text = self.model_run(prompt, img_path)
                    if not page_text.strip():
                        page_text = "[No text extracted from this page]"
                    bill_texts.append(f"--- Page {i+1} ---\n{page_text}")

                bill_text = "\n\n".join(bill_texts)
            except Exception as e:
                return f"❌ PDF processing failed: {e}"
            finally:
                # Clean up temporary images
                for img_path in image_paths:
                    try:
                        os.remove(img_path)
                    except:
                        pass
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
        else:
            # Process single image
            print("📝 Extracting text...")
            bill_text = self.model_run(prompt, image_path)

        if not bill_text or not bill_text.strip():
            return "❌ No text extracted!"

        print("🧠 Parsing with AI...")
        parsed_data = self.parse_bill_with_llm(bill_text)
        print(f"ParsedData:\n{parsed_data}")
        if not parsed_data:
            return "❌ Parsing failed!"

        self.current_bill = parsed_data
        self.bill_text = bill_text

        # chunk & compute embeddings (semantic Q&A)
        print("🔍 Creating text chunks...")
        self.chunks = self.chunk_text(bill_text, chunk_size=400, overlap=50)
        if not self.chunks:
            self.chunks = [bill_text]

        print(f"🔢 {len(self.chunks)} chunks created. Computing embeddings...")
        self.compute_chunk_embeddings()

        return "✅ Bill processed successfully!"

    def _preprocess_text(self, text: str):
        """Preprocess text for BM25: lowercase, remove punctuation, remove stopwords."""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        return tokens

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray):
        "Compute cosine similarity between 1D a and 2D b (b is list of vectors)."
        if a.ndim == 1:
            a = a.reshape(1, -1)
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_norm, b_norm.T).squeeze(0)  # shape (n_b,)

    def semantic_search(self, query: str, top_k: int = 3, alpha: float = 0.7):
        """
        Hybrid search combining BM25 (keyword) and semantic (embedding) scores.
        alpha: weight for semantic similarity (0.0 = BM25 only, 1.0 = semantic only)
        Returns list of tuples: (chunk_text, combined_score, chunk_index)
        """
        if not self.chunks:
            return []

        # Always compute semantic scores if available
        semantic_scores = np.zeros(len(self.chunks))
        if self.chunk_embeddings is not None and self.sentence_model is not None:
            try:
                q_emb = self.sentence_model.encode([query], convert_to_numpy=True)[0]
                q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
                semantic_scores = self._cosine_sim(q_emb, self.chunk_embeddings)
            except Exception as e:
                print(f"⚠️ Semantic search failed: {e}")

        # Compute BM25 scores if available
        bm25_scores = np.zeros(len(self.chunks))
        if self.bm25 is not None:
            try:
                query_tokens = self._preprocess_text(query)
                bm25_scores = self.bm25.get_scores(query_tokens)
            except Exception as e:
                print(f"⚠️ BM25 search failed: {e}")

        # Normalize scores to [0, 1] range
        def normalize(scores):
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score - min_score < 1e-10:  # Avoid division by zero
                return scores
            return (scores - min_score) / (max_score - min_score + 1e-10)

        norm_semantic = normalize(semantic_scores)
        norm_bm25 = normalize(bm25_scores)

        # Combine scores with weight alpha
        combined_scores = alpha * norm_semantic + (1 - alpha) * norm_bm25

        # Get top indices (descending order)
        top_idx = np.argsort(-combined_scores)[:top_k]
        results = [(self.chunks[i], float(combined_scores[i]), int(i)) for i in top_idx]
        return results


    def _clean_json(self, text: str) -> str:
      if text is None:
          return ""
      text = text.replace('\\r\\n', '\n').replace('\\n', '\n').strip()
      text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
      text = re.sub(r"\n?```$", "", text, flags=re.IGNORECASE)
      return text.strip()

    def _basic_repair(self, text: str) -> str:
      t = text
      # common fixes (same idea as earlier)
      t = re.sub(r'"\s+"', r'"', t)
      t = re.sub(r'"\s+""', r'""', t)
      t = re.sub(r':\s*"\s*"\s*([^"\n\r]+)"', r': "\1"', t)
      t = re.sub(r':\s*"\s*([^"\n\r]+)"', r': "\1"', t)  # keep trying to remove stray quotes
      t = re.sub(r'":\s*"\s+([^"\n\r]+)"', r'": "\1"', t)
      t = re.sub(r"\'([^\']*)\'", r'"\1"', t)
      t = re.sub(r',\s*(\}|\])', r'\1', t)
      t = re.sub(r'"\s*\}\s*\s*\{', r'"\n},\n{', t)
      t = ''.join(ch for ch in t if ch == '\n' or (31 < ord(ch) < 127))
      t = re.sub(r'(?m)^(\s*)([A-Za-z0-9_\-]+)\s*:', r'\1"\2":', t)
      t = re.sub(r'"\s+([^"]+?)\s+"', lambda m: f'"{m.group(1).strip()}"', t)
      # don't auto-append braces/brackets here — leave to the more aggressive routine
      return t.strip()

    def _balance_closers(self, s: str) -> str:
      # Add minimal closers to match opens
      open_braces = s.count('{')
      close_braces = s.count('}')
      open_brackets = s.count('[')
      close_brackets = s.count(']')
      if open_braces > close_braces:
          s = s + ('}' * (open_braces - close_braces))
      if open_brackets > close_brackets:
          s = s + (']' * (open_brackets - close_brackets))
      return s

    def safe_load_json_recover(self, raw_text: str, debug: bool = False, max_trim_chars: int = 3000):
      """
      Attempt to clean/repair LLM-produced JSON and recover a Python object.
      Strategy:
        1) Clean and do basic regex repairs.
        2) Try json.loads.
        3) If it fails, attempt balancing quotes/braces/brackets and try again.
        4) If still fails, progressively trim from the end (up to `max_trim_chars`) and try balancing + loads.
      Returns: Python object (dict/list) or raises ValueError with repaired snippet for inspection.
      """
      cleaned = self._clean_json(raw_text)
      repaired = self._basic_repair(cleaned)

      if debug:
          print("=== Initial repaired ===")
          print(repaired)
          print("=== Trying json.loads ===")

      # 1) Try direct load after basic repair
      try:
          return json.loads(repaired)
      except json.JSONDecodeError as e:
          pass

      # 2) If odd number of double-quotes, try closing the last quote
      if repaired.count('"') % 2 == 1:
          cand = repaired + '"'
          try:
              return json.loads(self._balance_closers(cand))
          except Exception:
              # continue to trimming attempts
              pass

      # 3) Try adding minimal closers and reloading
      cand = self._balance_closers(repaired)
      try:
          return json.loads(cand)
      except Exception:
          pass

      # 4) Progressive trimming: remove trailing characters (one by one or in small chunks)
      #    and try to parse the prefix + balanced closers.
      L = len(repaired)
      # we'll try trimming up to max_trim_chars, in steps (bigger steps at first)
      step = 1
      trimmed = None
      for trim in range(0, min(max_trim_chars, L), step):
          if trim == 0:
              candidate = repaired
          else:
              candidate = repaired[:L - trim]

          # If candidate ends in a partial token like ' "qty": 3, "amount":', remove a trailing incomplete token:
          # remove trailing sequences after the last '}' or ']' if they exist
          last_close = max(candidate.rfind('}'), candidate.rfind(']'))
          if last_close != -1 and last_close > len(candidate) - 200:
              # keep up to last_close (close object/array)
              candidate = candidate[:last_close+1]

          # Close any open quotes (best-effort)
          if candidate.count('"') % 2 == 1:
              candidate = candidate + '"'

          # Balance braces/brackets
          candidate = self._balance_closers(candidate)

          try:
              parsed = json.loads(candidate)
              if debug:
                  print(f"Recovered by trimming {trim} chars.")
              return parsed
          except Exception:
              # increase step size after initial few tries to speed up
              if trim < 50:
                  step = 1
              elif trim < 200:
                  step = 5
              else:
                  step = 20
              continue

      # If all attempts fail, raise with helpful diagnostic including the best-effort repaired text
      # Provide truncated snippet to avoid enormous message
      best_snippet = repaired[:4000] + ("... (truncated)" if len(repaired) > 4000 else "")
      msg = (
          "Failed to parse JSON after attempted repairs and progressive trimming.\n\n"
          "A best-effort repaired snippet is shown below (inspect to decide next action):\n\n"
          f"{best_snippet}\n\n"
          "Common fixes: ask the LLM to re-output only the JSON inside a single ```json``` codeblock, "
          "increase the model `max_tokens` for longer outputs, or detect why the output was truncated."
      )
      raise ValueError(msg)

    # ------ Parsing (LLM or hardcoded) ------
    def parse_bill_with_llm(self, text: str):
        "Parse bill text into structured JSON."
        print(f"Parsing with LLM...")
        # If you want to call Groq LLM, you might do something like:
        if self.client is not None:
            print(f"Cient found.")
            prompt = f"""
                Extract structured data from this bill text as JSON:
                {text}

                Required fields:
                - invoice_number, invoice_date, due_date, po_number
                - bill_to (name, address, phone)
                - ship_to (name, address, phone)
                - items (qty, description, unit_price, amount)
                - subtotal, tax, total
                - company_name, company_address

                Return ONLY valid JSON with these fields. Use empty strings for missing data.
                """
            print(f"\n\nPROMPT:\n{prompt}\n\n")
            response = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.1-8b-instant",
                        temperature=0.1,
                        max_tokens=1024
                    )
            response_content = response.choices[0].message.content

            try:
                parsed = self.safe_load_json_recover(response_content, debug=True)
                print("Parsed OK:", parsed)
                return parsed
            except ValueError as e:
                print("Could not parse. Inspect repaired output:")
                print(str(e))
                return {}

    # ---------------- answering queries ----------------
    def answer_query(self, query: str, top_k: int = 3):
        if not self.current_bill:
            return "❌ Process a bill first!"

        query_lower = query.lower()
        if "total" in query_lower:
            return f"💰 Total: ${self.current_bill.get('total', 'N/A')}"
        if "invoice number" in query_lower or "invoice #" in query_lower or "invoice" == query_lower.strip():
            return f"📋 Invoice: {self.current_bill.get('invoice_number', 'N/A')}"
        if "date" in query_lower:
            return f"📅 Date: {self.current_bill.get('invoice_date', 'N/A')}"
        if "items" in query_lower:
            items = self.current_bill.get("items", [])
            if items:
                df = pd.DataFrame(items)
                return f"🛒 Items:\n{tabulate(df, headers='keys', tablefmt='grid')}"
            return "ℹ️ No items found"

        # fallback to semantic retrieval
        retrieved = self.semantic_search(query, top_k=top_k)
        if not retrieved:
            return "ℹ️ No relevant content found in the bill."
        # print(f"\nRETRIEVED DATA\n{retrieved}")

        if self.client is not None:
            context_text = "\n\n---\n\n".join([f"Chunk {idx} (score {score:.4f}):\n{text}" for text, score, idx in retrieved])
            prompt = (
                f"Use ONLY the context below (do NOT hallucinate). "
                f"Extract an answer to the question. If information is not present, say 'Not found'.\n\n"
                f"CONTEXT:\n{context_text}\n\n"
                f"QUESTION: {query}\n\n"
                f"Answer concisely based ONLY on the context above:"
            )
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=0.0,
                    max_tokens=256
                )
                return response.choices[0].message.content
            except Exception as e:
                # fallback to returning chunks
                return f"⚠️ LLM query failed: {e}\n\nTop relevant text:\n\n" + "\n\n---\n\n".join([t for t, s, i in retrieved])

        # If no LLM, return the top chunks concatenated with scores
        best_text = "\n\n---\n\n".join([f"(score {s:.4f})\n{t}" for t, s, i in retrieved])
        return f"🔎 Top relevant bill text (best {len(retrieved)} chunks):\n\n{best_text}"

    # ------ Export ------
    def export_to_json(self, filename: str = None):
        "Export the current bill to a JSON file."
        if not self.current_bill:
            return "❌ No bill data available!"

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bill_{timestamp}.json"

        try:
            with open(filename, "w") as f:
                json.dump(self.current_bill, f, indent=2)
        except Exception as e:
            return f"❌ Failed to export: {e}"
        return f"✅ Exported to {filename}"

    # ------ Chat ------
    def chat(self, message: str):
        "Simple chat handler. Uses Groq if available for richer replies."
        msg_lower = message.lower().strip()

        if msg_lower in ("hi", "hello", "hey"):
            return "👋 Hello! I'm your bill assistant. I can help you extract data from bills, answer questions, and export information."

        if "help" in msg_lower:
            help_text = "💡 I can:\n1. Process bill images and PDFs\n2. Answer questions about bills\n3. Export data to JSON\n4. General chat"
            if self.current_bill:
                help_text += "\n\n✅ Current bill loaded! Ask about totals, dates, items, or custom questions."
            else:
                help_text += "\n\n⚠️ No bill processed yet. Please process a bill first to ask questions about it."
            return help_text

        # Handle queries when bill is loaded
        if self.current_bill is not None:
            return self.answer_query(message, top_k=3)

        # No bill loaded - provide helpful guidance without LLM calls
        if any(keyword in msg_lower for keyword in ["bill", "invoice", "receipt", "document", "pdf", "image", "process", "upload", "load", "scan"]):
            return "📎 Please process a bill first using option 1 in the main menu. I can handle images and PDFs!"

        if any(keyword in msg_lower for keyword in ["thank", "bye", "exit", "goodbye"]):
            return "👋 You can exit chat mode anytime by typing 'exit' or 'quit'."

        # General fallback when no bill is loaded
        return "🤖 I'm ready to help with bills! Please process a bill first (option 1), then ask questions like 'What's the total?' or 'Show items'. Type 'help' for options."

    def chat_loop(self):
        "Continuous chat loop until user exits."
        print("\n💬 Chat mode activated (type 'exit' or 'quit' to return to main menu)")
        print("🤖 Assistant: Hello! I'm your bill assistant. How can I help you today?")

        while True:
            message = input("You: ").strip()

            if message.lower() in ('exit', 'quit', 'bye', 'goodbye'):
                print("👋 Exiting chat mode...")
                break

            response = self.chat(message)
            print(f"🤖 Assistant: {response}")

    def run_cli(self):
        """Interactive command-line menu."""
        self.print_device_info()
        print("\n" + "=" * 50)
        print("BILL ASSISTANT")
        print("=" * 50)

        print("\n1. 📸 Process bill image/PDF")
        print("2. 💾 Export to JSON")
        print("3. 💬 Chat")
        print("4. 🚪 Exit")

        while True:
            choice = input("\nSelect option (1-4): ").strip()

            if choice == "1":
                torch.cuda.empty_cache()
                gc.collect()
                img_path = input("Enter image/PDF path (or URL): ").strip()
                result = self.process_bill(img_path)
                print(result)
                if self.current_bill:
                    print("\n📊 Bill Summary:")
                    print(f"Invoice: {self.current_bill.get('invoice_number', 'N/A')}")
                    print(f"Date: {self.current_bill.get('invoice_date', 'N/A')}")
                    print(f"Total: ${self.current_bill.get('total', 'N/A')}")
                print("\n" + "=" * 50)
                print("BILL ASSISTANT")
                print("=" * 50)
                print("\n1. 📸 Process bill image/PDF")
                print("2. 💾 Export to JSON")
                print("3. 💬 Chat")
                print("4. 🚪 Exit")

            elif choice == "2":
                filename = input("Enter filename (or press Enter for default): ").strip()
                result = self.export_to_json(filename if filename else None)
                print(result)

            elif choice == "3":
                self.chat_loop()  # Call the new continuous chat loop
                # Show menu again after exiting chat
                print("\n" + "=" * 50)
                print("BILL ASSISTANT")
                print("=" * 50)
                print("\n1. 📸 Process bill image/PDF")
                print("2. 💾 Export to JSON")
                print("3. 💬 Chat")
                print("4. 🚪 Exit")

            elif choice == "4":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice!")

# # If run as script, start the CLI3
# if __name__ == "__main__":
#     assistant = BillAssistant(use_colab_secrets=True)
#     assistant.run_cli()

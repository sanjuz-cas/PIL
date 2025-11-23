import numpy as np
import time
import logging
from sentence_transformers import SentenceTransformer
from jinja2 import Template
from duckduckgo_search import DDGS
from googleapiclient.discovery import build
from app.core.config import settings
from app.core.memory import MemoryLayer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("indxai.engine")


class BrowserTool:
    def __init__(self):
        self.google_key = (
            settings.GOOGLE_API_KEY if hasattr(settings, "GOOGLE_API_KEY") else None
        )
        self.google_cse_id = (
            settings.GOOGLE_CSE_ID if hasattr(settings, "GOOGLE_CSE_ID") else None
        )

    def search(self, query: str) -> str:
        # Tier 1: Google
        if self.google_key and self.google_cse_id:
            try:
                service = build("customsearch", "v1", developerKey=self.google_key)
                res = (
                    service.cse().list(q=query, cx=self.google_cse_id, num=2).execute()
                )
                if "items" in res:
                    return " ".join([item["snippet"] for item in res["items"]])
            except:
                pass

        # Tier 2: DuckDuckGo
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=2, backend="html"))
                if results:
                    return " ".join([r["body"] for r in results])
        except:
            pass
        return ""


class PILVAEDecoder:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.weights = {}

    def train_analytical(self, X_embeddings):
        if len(X_embeddings) < 2:
            return
        try:
            self.mean = np.mean(X_embeddings, axis=0)
            self.std = np.std(X_embeddings, axis=0) + 1e-6
            X = (X_embeddings - self.mean) / self.std

            u, s, vt = np.linalg.svd(X.T, full_matrices=False)
            W_enc = u[:, : self.latent_dim]

            Z = X @ W_enc
            lambda_reg = 0.1
            Z_inv = np.linalg.inv(Z.T @ Z + lambda_reg * np.eye(self.latent_dim)) @ Z.T
            W_dec = Z_inv @ X

            self.weights = {"encoder": W_enc, "decoder": W_dec}
            logger.info(f"✅ Brain retrained on {len(X_embeddings)} vectors.")
        except Exception as e:
            logger.error(f"Math Error: {e}")


class IndxAI_OS:
    def __init__(self):
        logger.info("🚀 Booting indxai OS...")
        self.memory = MemoryLayer(embedding_dim=settings.EMBEDDING_DIM)
        self.browser = BrowserTool()
        self.pil_vae = PILVAEDecoder(latent_dim=settings.LATENT_DIM)

        try:
            self.transformer = SentenceTransformer(settings.TRANSFORMER_MODEL)
        except:
            self.transformer = None

        # FIX: Use correct method to load vectors
        vecs = self.memory.get_all_vectors()
        if len(vecs) > 0:
            self.pil_vae.train_analytical(vecs)

    def encode(self, text):
        if self.transformer:
            return self.transformer.encode(text)
        return np.random.randn(settings.EMBEDDING_DIM)

    def learn(self, text_blob: str, source: str = "web"):
        sentences = [s for s in text_blob.split(".") if len(s) > 20]
        for s in sentences:
            self.memory.add(s, self.encode(s), source)

        # FIX: Use correct method to retrieve vectors
        all_vecs = self.memory.get_all_vectors()
        if len(all_vecs) > 0:
            self.pil_vae.train_analytical(all_vecs)
        return len(sentences)

    def run_query(self, query: str):
        start = time.time()
        query_vec = self.encode(query)

        docs = self.memory.retrieve(query_vec, top_k=3)
        best_score = docs[0]["score"] if docs else 0

        # Privacy Guardrail
        is_personal = any(
            w in query.lower() for w in ["my", "email", "slack", "private", "file"]
        )
        knowledge_gap = best_score < 0.45

        context = ""
        source_label = "MEMORY"

        if is_personal:
            if best_score > 0.35:
                context = " ".join([d["text"] for d in docs])
            else:
                context = (
                    "No matching private records found. (Run connectors to ingest data)"
                )

        elif knowledge_gap:
            logger.info(f"🧠 Searching Web for: {query}")
            web_result = self.browser.search(query)

            if web_result:
                context = web_result
                source_label = "LIVE WEB"
                self.learn(web_result, source="auto_web")
            else:
                context = " ".join([d["text"] for d in docs])
                if not context:
                    context = "I could not find info on the web or in memory."
        else:
            context = " ".join([d["text"] for d in docs])

        response = f"[{source_label}]: {context}"
        return response, (time.time() - start) * 1000

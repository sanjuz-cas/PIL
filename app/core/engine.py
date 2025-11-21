import numpy as np
import time
import random
from sentence_transformers import SentenceTransformer
from jinja2 import Template
from app.core.config import settings
from app.core.memory import MemoryLayer
from app.core.tools import BrowserTool


class PILVAEDecoder:
    """The 'Writing' Brain: Gradient-Free Generator."""

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.weights = {}
        self.mean = None
        self.std = None

    def train_analytical(self, X_embeddings):
        # Check if we have enough data to train
        if len(X_embeddings) < 2:
            # Not enough data for covariance, use identity init
            self.weights["encoder"] = np.eye(X_embeddings.shape[1], self.latent_dim)
            self.weights["decoder"] = np.eye(self.latent_dim, X_embeddings.shape[1])
            self.mean = np.zeros(X_embeddings.shape[1])
            self.std = np.ones(X_embeddings.shape[1])
            return

        # 1. Normalize
        self.mean = np.mean(X_embeddings, axis=0)
        self.std = np.std(X_embeddings, axis=0) + 1e-6
        X = (X_embeddings - self.mean) / self.std

        # 2. Encoder (PCA-like)
        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][: self.latent_dim]
        W_enc = eigvecs[:, idx]
        self.weights["encoder"] = W_enc

        # 3. Decoder (Pseudoinverse / Ridge Regression)
        Z = X @ W_enc
        lambda_reg = 0.1
        Z_prime = np.linalg.inv(Z.T @ Z + lambda_reg * np.eye(self.latent_dim)) @ Z.T
        W_dec = Z_prime @ X
        self.weights["decoder"] = W_dec


class IndxAI_OS:
    """Main Operating System Class"""

    def __init__(self):
        print("🚀 Booting indxai Hybrid Engine...")

        self.memory = MemoryLayer(embedding_dim=settings.EMBEDDING_DIM)
        self.browser = BrowserTool()
        self.pil_vae = PILVAEDecoder(latent_dim=settings.LATENT_DIM)

        try:
            self.transformer = SentenceTransformer(settings.TRANSFORMER_MODEL)
        except Exception as e:
            print(f"⚠️ Model load failed: {e}")
            self.transformer = None

        self.mode = "assistant"
        self._seed_knowledge()

    def encode(self, text):
        if self.transformer:
            return self.transformer.encode(text)
        return np.random.randn(settings.EMBEDDING_DIM)

    def _seed_knowledge(self):
        """
        Seed with Startup Info AND General Fallbacks.
        This prevents 'Empty Brain' syndrome.
        """
        facts = [
            "indxai is a startup building gradient-free generative AI.",
            "PIL-VAE is 17x faster than GANs and 900x faster than Diffusion.",
            "We target edge devices and enterprise on-premise servers.",
            "The hybrid engine uses a Mini-Transformer for reading and PIL for writing.",
        ]

        vectors = []
        for f in facts:
            vec = self.encode(f)
            vectors.append(vec)
            self.memory.add(f, vec, {"type": "core_knowledge"})

        if len(vectors) > 0:
            self.pil_vae.train_analytical(np.array(vectors))

    def run_query(self, user_input: str):
        start_time = time.time()

        # 1. Encode
        query_vec = self.encode(user_input)

        # 2. Retrieval (RAG)
        docs = self.memory.retrieve(query_vec)

        # CHECK: Is the retrieved memory actually relevant?
        # In a real system we check cosine score. Here we'll trust the browser more.
        internal_context = " ".join([d["text"] for d in docs]) if docs else ""

        # 3. Browser Check (Aggressive)
        # Always try browser if the query looks like a question about the world
        live_data = ""
        triggers = [
            "who",
            "what",
            "where",
            "when",
            "price",
            "news",
            "current",
            "president",
            "weather",
        ]

        if any(k in user_input.lower() for k in triggers):
            try:
                live_data = self.browser.search(user_input)
            except:
                live_data = "Web search failed (Cloud IP Blocked)."

        # 4. Logic Routing
        # If we found live data, use THAT. Ignore internal memory about "edge devices".
        final_context = ""

        if live_data and "No relevant" not in live_data:
            final_context = f"Live Web Data: {live_data}"
        else:
            # Fallback to internal memory only if it seems relevant
            # (Simple heuristic: if query contains 'indxai' or 'pil', use memory)
            if (
                "indxai" in user_input.lower()
                or "pil" in user_input.lower()
                or "fast" in user_input.lower()
            ):
                final_context = f"Internal Database: {internal_context}"
            else:
                # If we know nothing and browser failed
                final_context = "I am a specialized Edge AI trained on indxai technology. I don't have general world knowledge installed in this demo version."

        # 5. Generation Template
        if self.mode == "wearable":
            response = f"{{ 'answer': '{final_context[:100]}...', 'latency': 'low' }}"
        else:
            tmpl_str = """
            [Analysis]: Processing query...
            [Context]: {{final_context}}
            [Response]: {{final_context}}
            """
            t = Template(tmpl_str)
            response = t.render(final_context=final_context)

        response = response.replace("\n", " ").strip()

        # Log
        self.memory.add_history("user", user_input)
        self.memory.add_history("ai", response)

        latency = (time.time() - start_time) * 1000
        return response, latency

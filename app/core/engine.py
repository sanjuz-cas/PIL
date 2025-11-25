import time
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from app.core.pil_vae import PILVAE
from app.core.gmail_tool import GmailTool
from app.core.config import settings
from app.core.memory import MemoryLayer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndxAI_OS:
    def __init__(self):
        logger.info("Initializing PIL-VAE Hybrid Engine...")
        # Load Embedding Model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize PIL-VAE
        # MiniLM has 384 dimensions
        self.vae = PILVAE(input_dim=384, latent_dim=24, hidden_dim=128)

        # Initialize Tools
        self.gmail = GmailTool()

        # Memory Stores
        self.memory = MemoryLayer(embedding_dim=384)
        self.memory_texts = []
        self.memory_embeddings = None  # Numpy array

        # Pre-load memory from DB
        self._hydrate_memory()

        # Compatibility
        self.mode = "assistant"

        logger.info("Engine Ready.")

    def _hydrate_memory(self):
        """Load existing vectors from DB into VAE training set"""
        vecs = self.memory.get_all_vectors()
        if len(vecs) > 0:
            self.memory_embeddings = vecs
            # We don't have the raw texts easily accessible in bulk from the current MemoryLayer
            # implementation without a new method, but for VAE training we only need vectors.
            # For retrieval, we will query the DB.
            self.vae.fit(self.memory_embeddings)

    def _extract_keywords(self, query):
        """Basic keyword extraction for better search queries"""
        stopwords = {
            "my",
            "last",
            "gmail",
            "email",
            "emails",
            "message",
            "messages",
            "from",
            "about",
            "check",
            "search",
            "find",
            "in",
            "the",
            "for",
            "to",
            "on",
            "with",
            "show",
            "me",
            "get",
            "read",
            "fetch",
        }
        words = query.lower().split()
        keywords = [w for w in words if w not in stopwords]
        return " ".join(keywords)

    def search_web(self, query, max_results=5):
        """Robust web search with fallback."""
        results = []
        try:
            # Try Lite backend first (faster)
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(query, max_results=max_results, backend="lite")
                )
        except Exception as e:
            logger.warning(f"Lite backend failed: {e}. Switching to default.")
            try:
                # Fallback to default/html
                with DDGS() as ddgs:
                    results = list(
                        ddgs.text(query, max_results=max_results, backend="html")
                    )
            except Exception as e2:
                logger.error(f"All search backends failed: {e2}")

        return results

    def update_memory(self, new_texts, source="web"):
        """Embeds new texts and updates the VAE."""
        if not new_texts:
            return

        # 1. Embed Text
        new_embs = self.embedder.encode(new_texts)

        # 2. Update Storage (DB)
        for text, emb in zip(new_texts, new_embs):
            self.memory.add(text, emb, source)

        # 3. Update Local Cache & VAE
        if self.memory_embeddings is None:
            self.memory_embeddings = new_embs
        else:
            self.memory_embeddings = np.vstack([self.memory_embeddings, new_embs])

        # 4. Fit VAE on ALL memory (Closed-form is fast)
        self.vae.fit(self.memory_embeddings)

    def learn_new_data(self, text_blob: str):
        """Compatibility wrapper for endpoint."""
        # Split by sentences roughly
        sentences = [s.strip() for s in text_blob.split(".") if len(s.strip()) > 10]
        if sentences:
            self.update_memory(sentences, source="user_training")

    def get_reasoning_response(self, query_vec, query_text, forced_context=None):
        """
        V3 Reasoning Logic:
        Latent -> Reconstruct -> k-NN -> Composition
        """
        # 1. VAE "Reasoning" Step
        z = self.vae.encode(query_vec)
        e_gen = self.vae.decode(z)

        # 2. Nearest-Neighbor Retrieval (Cosine Similarity)
        # We use the DB retrieval for persistence
        docs = self.memory.retrieve(e_gen, top_k=3)

        top_texts = []
        top_scores = []

        # Priority 1: Forced Context (Live Data)
        if forced_context:
            # We trust live data 100%
            for text in forced_context[:3]:  # Limit to top 3 live results
                top_texts.append(text)
                top_scores.append(1.0)

        # Priority 2: Memory Retrieval
        if docs:
            for d in docs:
                # Avoid duplicates
                if d["text"] not in top_texts:
                    top_texts.append(d["text"])
                    top_scores.append(d["score"])

        if not top_texts:
            return "I couldn't find any relevant information in my memory."

        # Limit to top 5 total for display
        top_texts = top_texts[:5]
        top_scores = top_scores[:5]

        # 3. Reasoning Composition
        combined_text = " ".join(top_texts)
        words = [w for w in combined_text.split() if len(w) > 4]
        from collections import Counter

        common = Counter(words).most_common(3)
        concepts = [c[0] for c in common]

        # Structure the response
        response = f"**Analysis**: Based on the latent reconstruction, the key concepts are {', '.join(concepts)}.\n\n"
        response += "**Evidence**:\n"
        for txt, score in zip(top_texts, top_scores):
            # Show full text if reasonable, otherwise truncate at sentence boundary
            display_txt = txt.replace("\n", " ")
            if len(display_txt) > 300:
                # Find last period before 300 chars
                last_period = display_txt[:300].rfind(".")
                if last_period > 50:
                    display_txt = display_txt[: last_period + 1]
                else:
                    display_txt = display_txt[:300] + "..."

            response += f"- [{score:.2f}] {display_txt}\n"

        response += (
            f"\n**Conclusion**: {query_text} seems to be linked to these findings. "
        )
        if top_texts:
            # Use the first sentence of the top result for the conclusion
            first_result = top_texts[0].replace("\n", " ")
            first_sentence = first_result.split(".")[0] + "."
            response += f"The data suggests {first_sentence}"

        return response

    def run_query_generator(self, query: str):
        start = time.time()

        # 1. Intent Routing & Context Retrieval
        context_data = []
        source_label = "LIVE WEB"
        sources_metadata = []

        # Simple keyword detection for Gmail intent
        if (
            "gmail" in query.lower()
            or "email" in query.lower()
            or "inbox" in query.lower()
        ):
            source_label = "GMAIL"
            clean_q = self._extract_keywords(query)
            # GmailTool returns strings
            raw_results = self.gmail.get_relevant_emails(clean_q)
            context_data = raw_results

            # Create metadata for Gmail
            for i, txt in enumerate(raw_results):
                sources_metadata.append(
                    {
                        "title": f"Email Result {i + 1}",
                        "url": "https://mail.google.com",
                        "snippet": txt[:100] + "...",
                    }
                )
        else:
            # Default to Web Search
            # Returns list of dicts: {'title':..., 'href':..., 'body':...}
            raw_results = self.search_web(query)
            context_data = [r.get("body", "") for r in raw_results if "body" in r]

            for r in raw_results:
                sources_metadata.append(
                    {
                        "title": r.get("title", "Web Result"),
                        "url": r.get("href", "#"),
                        "snippet": r.get("body", "")[:100] + "...",
                    }
                )

        # 2. Update Memory & Train VAE
        if context_data:
            self.update_memory(context_data, source=source_label)

        if self.memory_embeddings is None:
            yield (
                json.dumps(
                    {
                        "type": "token",
                        "content": f"[{source_label}] No data found. Please check connections.",
                    }
                )
                + "\n"
            )
            # Send empty meta to close gracefully
            yield (
                json.dumps(
                    {
                        "type": "meta",
                        "stats": {"latency": "0ms"},
                        "sources": [],
                        "actions": [],
                    }
                )
                + "\n"
            )
            return

        # 3. Embed Query
        query_vec = self.embedder.encode(query)

        # 4. Generate Reasoning via VAE
        reasoned_text = self.get_reasoning_response(
            query_vec, query, forced_context=context_data
        )

        # 5. Conversational Formatting & Streaming
        final_output = (
            f"Hey, checking {source_label} for you...\n\n"
            f"{reasoned_text}\n\nHope that helps!"
        )

        # Stream character by character
        chunk_size = 3
        for i in range(0, len(final_output), chunk_size):
            chunk = final_output[i : i + chunk_size]
            yield json.dumps({"type": "token", "content": chunk}) + "\n"
            time.sleep(0.03)

        # Final stats & Metadata
        latency = (time.time() - start) * 1000
        yield (
            json.dumps(
                {
                    "type": "meta",
                    "stats": {"latency": f"{latency:.2f}ms"},
                    "sources": sources_metadata,
                    "actions": ["copy", "thumbs_up", "thumbs_down"],
                }
            )
            + "\n"
        )

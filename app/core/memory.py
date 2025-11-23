import sqlite3
import numpy as np
import time
from typing import List, Dict


class MemoryLayer:
    """
    Persistent Memory (SQLite + Hot Cache).
    """

    def __init__(self, db_path="brain.db", embedding_dim=384):
        self.db_path = db_path
        self.dim = embedding_dim
        self.history = []  # RAM Cache for UI
        self._init_db()
        self._load_history_from_disk()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    vector BLOB,
                    source TEXT,
                    timestamp REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    content TEXT,
                    timestamp REAL
                )
            """)

    def _load_history_from_disk(self):
        with sqlite3.connect(self.db_path) as conn:
            try:
                cursor = conn.execute(
                    "SELECT role, content FROM history ORDER BY id DESC LIMIT 50"
                )
                rows = cursor.fetchall()
                for row in reversed(rows):
                    self.history.append({"role": row[0], "content": row[1]})
            except Exception:
                pass

    def add(self, text: str, vector: np.ndarray, source: str = "auto"):
        with sqlite3.connect(self.db_path) as conn:
            vec_blob = vector.astype(np.float32).tobytes()
            conn.execute(
                "INSERT INTO knowledge (text, vector, source, timestamp) VALUES (?, ?, ?, ?)",
                (text, vec_blob, source, time.time()),
            )

    def retrieve(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict]:
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT text, vector FROM knowledge")
            for row in cursor:
                text, vec_blob = row
                vec = np.frombuffer(vec_blob, dtype=np.float32)
                try:
                    score = np.dot(vec, query_vector) / (
                        (np.linalg.norm(vec) * np.linalg.norm(query_vector)) + 1e-8
                    )
                except:
                    score = 0
                results.append({"text": text, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_all_vectors(self):
        """
        Crucial Fix: Fetches vectors from DB instead of a list.
        """
        vectors = []
        with sqlite3.connect(self.db_path) as conn:
            try:
                cursor = conn.execute("SELECT vector FROM knowledge")
                for row in cursor:
                    vec = np.frombuffer(row[0], dtype=np.float32)
                    vectors.append(vec)
            except Exception:
                return np.empty((0, self.dim))

        if len(vectors) > 0:
            return np.array(vectors)
        else:
            return np.empty((0, self.dim))

    def add_history(self, role, content):
        self.history.append({"role": role, "content": content})
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO history (role, content, timestamp) VALUES (?, ?, ?)",
                (role, content, time.time()),
            )

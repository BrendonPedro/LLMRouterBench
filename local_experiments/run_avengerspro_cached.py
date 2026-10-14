"""
Run the official AvengersPro SimpleClusterRouter with persistent
query-embedding caching.

Scientific behavior is unchanged:
- same official EmbeddingGenerator
- same normalization
- same K-means
- same cluster rankings
- same routing
- same evaluation

The only addition is SQLite persistence of raw embedding vectors so
identical query/model embeddings do not need to be requested again.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

import baselines.AvengersPro.simple_cluster_router as official


CACHE_DB = Path(
    "baselines/AvengersPro/cache/"
    "gte_qwen2_7b_embeddings.sqlite"
)


class CachedSimpleClusterRouter(official.SimpleClusterRouter):

    def __init__(self, config):
        super().__init__(config)

        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)

        self._cache_model = getattr(
            self.embedder,
            "model_name",
            config.embedding_model,
        )

        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

        self._init_embedding_cache()

        self.logger.info(
            f"Persistent embedding cache: {CACHE_DB}"
        )
        self.logger.info(
            f"Embedding cache model: {self._cache_model}"
        )

    def _connect_cache(self):
        conn = sqlite3.connect(
            CACHE_DB,
            timeout=30,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_embedding_cache(self):
        with self._connect_cache() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    text TEXT,
                    embedding TEXT NOT NULL,
                    created_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (text_hash, model)
                )
                """
            )

    def _cache_get(self, query):
        text_hash = hashlib.sha256(
            query.encode("utf-8")
        ).hexdigest()

        with self._connect_cache() as conn:
            row = conn.execute(
                """
                SELECT embedding
                FROM embeddings
                WHERE text_hash = ?
                  AND model = ?
                """,
                (text_hash, self._cache_model),
            ).fetchone()

        if row is None:
            return None

        return np.asarray(
            json.loads(row[0]),
            dtype=float,
        )

    def _cache_put(self, query, embedding):
        text_hash = hashlib.sha256(
            query.encode("utf-8")
        ).hexdigest()

        payload = json.dumps(
            np.asarray(embedding, dtype=float).tolist()
        )

        with self._connect_cache() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (text_hash, model, text, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (
                    text_hash,
                    self._cache_model,
                    query,
                    payload,
                ),
            )

    def _get_embedding_batch(self, queries_batch):
        embeddings = []

        for query in queries_batch:

            embedding = self._cache_get(query)

            if embedding is not None:
                with self._cache_lock:
                    self._cache_hits += 1

            else:
                # IMPORTANT:
                # This is the original AvengersPro embedding path.
                embedding_output = (
                    self.embedder.generate_embedding(query)
                )

                embedding = np.asarray(
                    embedding_output.embeddings,
                    dtype=float,
                )

                if embedding.size == 0:
                    raise RuntimeError(
                        "Received empty embedding from generator"
                    )

                self._cache_put(query, embedding)

                with self._cache_lock:
                    self._cache_misses += 1

            # Preserve official AvengersPro behavior:
            # training vectors are raw here; test vectors are
            # transformed with the fitted normalizer.
            if self.normalizer:
                embedding = self.normalizer.transform(
                    [embedding]
                )[0]

            embeddings.append(embedding)

        return embeddings

    def run_routing(self):
        results = super().run_routing()

        print("\n" + "=" * 55)
        print("PERSISTENT EMBEDDING CACHE")
        print("=" * 55)
        print(f"Cache database : {CACHE_DB}")
        print(f"Cache hits     : {self._cache_hits}")
        print(f"API/cache miss : {self._cache_misses}")
        print("=" * 55)

        return results


# Replace only the class instantiated by the official main().
# All CLI parsing, configuration, training, routing and result
# serialization continue to use the repository's official code.
official.SimpleClusterRouter = CachedSimpleClusterRouter


if __name__ == "__main__":
    official.main()

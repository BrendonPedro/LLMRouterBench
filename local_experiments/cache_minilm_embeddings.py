import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CACHE = Path(
    "local_experiments/cache/"
    "minilm_l6_v2_embeddings.sqlite"
)

SEEDS = list(range(42, 52))


def collect_queries():
    queries = {}

    for seed in SEEDS:
        base = Path(
            f"baselines/AvengersPro/data/"
            f"small_models_seed_{seed}/"
            f"seed{seed}_split0.7"
        )

        for split in ["train.jsonl", "test.jsonl"]:
            path = base / split

            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    q = row["query"]
                    queries[q] = None

    return list(queries.keys())


def create_cache():
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(CACHE)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            text_hash TEXT NOT NULL,
            model TEXT NOT NULL,
            text TEXT,
            embedding TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (text_hash, model)
        )
        """
    )

    conn.commit()
    return conn


def cached_texts(conn):
    rows = conn.execute(
        """
        SELECT text
        FROM embeddings
        WHERE model = ?
        """,
        (MODEL,),
    ).fetchall()

    return {row[0] for row in rows}


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1)

    return (
        (last_hidden_state * mask).sum(dim=1)
        / mask.sum(dim=1).clamp(min=1e-9)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Encode only this many missing queries.",
    )

    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    queries = collect_queries()

    print("Unique benchmark queries :", len(queries))
    print("Model                    :", MODEL)
    print("Cache                    :", CACHE)
    print("CPU threads              :", args.threads)

    conn = create_cache()

    already_cached = cached_texts(conn)

    missing = [
        q for q in queries
        if q not in already_cached
    ]

    print("Already cached           :", len(already_cached))
    print("Missing                  :", len(missing))

    if args.limit is not None:
        missing = missing[:args.limit]
        print("Encoding this run        :", len(missing))

    if not missing:
        print("Nothing to encode.")
        conn.close()
        return

    print("Loading tokenizer/model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Encoder parameters       :", f"{params:,}")

    completed = 0

    for start in range(
        0,
        len(missing),
        args.batch_size,
    ):
        batch_texts = missing[
            start:start + args.batch_size
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**encoded)

        embeddings = mean_pool(
            outputs.last_hidden_state,
            encoded["attention_mask"],
        )

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        embeddings = (
            embeddings
            .cpu()
            .numpy()
        )

        rows = []

        for text, embedding in zip(
            batch_texts,
            embeddings,
        ):
            text_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            rows.append(
                (
                    text_hash,
                    MODEL,
                    text,
                    json.dumps(
                        embedding.tolist()
                    ),
                )
            )

        conn.executemany(
            """
            INSERT OR REPLACE INTO embeddings
            (
                text_hash,
                model,
                text,
                embedding
            )
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )

        conn.commit()

        completed += len(batch_texts)

        print(
            f"Encoded {completed}/{len(missing)}",
            flush=True,
        )

    total_cached = conn.execute(
        """
        SELECT COUNT(*)
        FROM embeddings
        WHERE model = ?
        """,
        (MODEL,),
    ).fetchone()[0]

    conn.close()

    print()
    print("Completed.")
    print("Total cached embeddings  :", total_cached)
    print("Embedding dimension      :", embeddings.shape[1])


if __name__ == "__main__":
    main()

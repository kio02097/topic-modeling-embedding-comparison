"""Command-line experiment runner with isolated outputs and cache validation."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from datetime import datetime, timezone

from .embeddings import MODELS, POOLING, cache_key, encode_documents
from .preprocessing import CUSTOM_STOPWORDS, DATASETS, prepare_documents


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, default="bbc")
    p.add_argument("--model", choices=[*MODELS, "lda"], default="sbert")
    p.add_argument("--pooling", choices=[*POOLING, "all"], default="CX")
    p.add_argument("--topics", type=int, nargs="+", default=list(range(2, 25, 2)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--umap-seed", type=int, default=None, help="Omit for paper Table 2 (None).")
    p.add_argument("--umap-components", type=int, default=5)
    p.add_argument("--min-tokens", type=int, default=5)
    p.add_argument("--limit", type=int, help="First N documents, for smoke runs only.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, help="Defaults to SBERT=256, LLaMA=512 (provisional).")
    p.add_argument("--chunk-size", type=int, help="Optional notebook extension; not confirmed in paper.")
    p.add_argument("--chunk-overlap", type=int, default=64)
    p.add_argument("--device", help="cpu, cuda, cuda:0, etc.")
    p.add_argument("--model-revision", help="Pin a Hugging Face model commit.")
    p.add_argument("--dataset-revision", help="Pin a Hugging Face dataset commit.")
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p


def validate_args(args, p):
    for name in ("batch_size", "min_tokens", "umap_components"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if any(n < 2 for n in args.topics) or len(set(args.topics)) != len(args.topics):
        p.error("--topics must contain distinct integers >= 2")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if args.max_length is not None and args.max_length < 1:
        p.error("--max-length must be positive")
    if args.chunk_size is not None and not 0 <= args.chunk_overlap < args.chunk_size:
        p.error("require 0 <= chunk-overlap < chunk-size")


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    p = parser()
    args = p.parse_args()
    validate_args(args, p)
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.feature_extraction.text import CountVectorizer
    from wordcloud import STOPWORDS
    from .modeling import evaluate_grid

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    start = time.perf_counter()
    frame, fingerprint = prepare_documents(args.dataset, args.min_tokens, args.limit, args.dataset_revision)
    docs = frame["lemma_text"].tolist()
    # LDA retains its notebook-specific custom stopwords. BERTopic uses WordCloud + dataset additions.
    stopwords = ["say", "make", "mr"] if args.model == "lda" else sorted(set(STOPWORDS) | set(CUSTOM_STOPWORDS[args.dataset]))
    max_length = args.max_length or (256 if args.model == "sbert" else 512)
    if args.chunk_size and args.chunk_size > max_length:
        p.error("--chunk-size must not exceed --max-length")
    poolings = ["NA"] if args.model == "lda" else (list(POOLING) if args.pooling == "all" else [args.pooling])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_dir / f"{args.dataset}-{args.model}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ("bertopic", "sentence-transformers", "transformers", "torch", "datasets", "numpy", "scipy",
                 "pandas", "scikit-learn", "gensim", "umap-learn", "hdbscan", "spacy", "nltk", "wordcloud"):
        versions[name] = importlib.metadata.version(name)
    metadata = {"arguments": vars(args), "python": platform.python_version(), "packages": versions,
                "documents": len(docs), "dataset_fingerprint": fingerprint,
                "document_hash": cache_key(docs, {}), "stopwords": stopwords,
                "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cuda": torch.version.cuda, "pythonhashseed": os.getenv("PYTHONHASHSEED"),
                "max_length": max_length, "status": "running", "embeddings": {}}
    write_json(run_dir / "metadata.json", metadata)
    try:
        all_rows, all_keywords = [], []
        for pooling in poolings:
            embedding_start = time.perf_counter()
            vectors = None
            if args.model != "lda":
                settings = {"dataset": args.dataset, "model": MODELS[args.model], "pooling": pooling,
                            "max_length": max_length, "chunk_size": args.chunk_size,
                            "chunk_overlap": args.chunk_overlap, "revision": args.model_revision,
                            "packages": versions, "seed": args.seed, "device": metadata["device"],
                            "dtype": "float32", "batch_size": args.batch_size}
                # Unpinned model revisions bypass reusable cache to prevent stale model reuse.
                key = cache_key(docs, settings)
                cache_path = args.cache_dir / f"{args.dataset}-{args.model}-{pooling}-{key}.npy"
                cache_meta = cache_path.with_suffix(".json")
                cache_hit = bool(args.model_revision and cache_path.exists() and cache_meta.exists())
                if cache_hit:
                    vectors = np.load(cache_path, allow_pickle=False)
                    info = json.loads(cache_meta.read_text(encoding="utf-8"))
                    if info["key"] != key:
                        raise ValueError("Embedding cache metadata mismatch")
                    resolved_revision = info.get("resolved_revision")
                else:
                    vectors, resolved_revision = encode_documents(docs, args.model, pooling, args.batch_size,
                        max_length, args.device, args.chunk_size, args.chunk_overlap, args.model_revision)
                    np.save(cache_path, vectors, allow_pickle=False)
                    write_json(cache_meta, {"key": key, "settings": settings, "resolved_revision": resolved_revision})
                expected_dim = (384 if args.model == "sbert" else 2048) * sum(POOLING[pooling])
                if vectors.shape != (len(docs), expected_dim) or not np.isfinite(vectors).all():
                    raise ValueError("Embedding shape or finite-value check failed")
                metadata["embeddings"][pooling] = {"key": key, "shape": list(vectors.shape),
                    "cache_hit": cache_hit, "resolved_revision": resolved_revision,
                    "seconds": time.perf_counter() - embedding_start}
            vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1, 1),
                                          lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
            rows, keywords = evaluate_grid(docs, vectorizer, args.topics, args.seed, vectors,
                                          args.umap_seed, args.umap_components, args.model == "lda")
            identity = {"dataset": args.dataset, "model": args.model, "pooling": pooling}
            all_rows.extend({**identity, **row} for row in rows)
            all_keywords.extend({**identity, **row} for row in keywords)
            result = pd.DataFrame(all_rows)
            result.to_csv(run_dir / "metrics.csv", index=False)
            write_json(run_dir / "topics.json", all_keywords)
            write_json(run_dir / "metadata.json", metadata)
        summary = []
        for pooling, group in result.groupby("pooling", sort=False):
            item = {"pooling": pooling, "topic_counts_tested": group["nr_topics"].tolist()}
            for metric in ("coherence_c_v", "coherence_c_npmi", "topic_diversity"):
                valid = group.dropna(subset=[metric])
                item[metric] = {"mean": float(valid[metric].mean()) if len(valid) else None,
                    "valid_runs": len(valid), "best": valid.loc[valid[metric].idxmax()].to_dict() if len(valid) else None}
            summary.append(item)
        write_json(run_dir / "summary.json", summary)
        metadata["status"] = "complete"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["elapsed_seconds"] = time.perf_counter() - start
        write_json(run_dir / "metadata.json", metadata)
    print(f"Results: {run_dir.resolve()}")


if __name__ == "__main__":
    main()

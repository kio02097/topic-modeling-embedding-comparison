"""Explicit pooling configurations; no model-default pooling is used."""
import hashlib
import json

POOLING = {
    "C": (True, False, False), "X": (False, True, False),
    "M": (False, False, True), "CM": (True, False, True),
    "CX": (True, True, False), "XM": (False, True, True),
    "CXM": (True, True, True),
}
MODELS = {"sbert": "sentence-transformers/all-MiniLM-L6-v2", "llama": "meta-llama/Llama-3.2-1B"}


def cache_key(docs, settings):
    payload = json.dumps({"documents": docs, "settings": settings}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_text(text, tokenizer, size, overlap):
    """Notebook-compatible decode/re-encode chunks; optional, not paper default."""
    if not 0 <= overlap < size:
        raise ValueError("chunk overlap must be >= 0 and smaller than chunk size")
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks, lengths = [], []
    for start in range(0, max(len(ids), 1), size - overlap):
        piece = ids[start:start + size]
        chunk = tokenizer.decode(piece, skip_special_tokens=True)
        chunks.append(chunk)
        lengths.append(len(tokenizer.encode(chunk, add_special_tokens=False)))
        if start + size >= len(ids):
            break
    return chunks, lengths


def encode_documents(docs, model_name, pooling, batch_size, max_length, device=None,
                     chunk_size=None, chunk_overlap=64, revision=None):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer, models

    kwargs = {"revision": revision} if revision else {}
    model_args = dict(kwargs)
    if model_name == "llama":
        model_args["torch_dtype"] = torch.float32
    transformer = models.Transformer(MODELS[model_name], max_seq_length=max_length,
                                     model_args=model_args, tokenizer_args=kwargs,
                                     config_args=kwargs)
    if model_name == "llama":
        tokenizer = transformer.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        transformer.auto_model.config.pad_token_id = tokenizer.pad_token_id
        # C denotes the first token (BOS when present), not a learned CLS token.
    cls, maximum, mean = POOLING[pooling]
    pool = models.Pooling(transformer.get_word_embedding_dimension(),
                          pooling_mode_cls_token=cls, pooling_mode_max_tokens=maximum,
                          pooling_mode_mean_tokens=mean)
    encoder = SentenceTransformer(modules=[transformer, pool], device=device)
    if chunk_size is None:
        return encoder.encode(docs, batch_size=batch_size, convert_to_numpy=True,
                              show_progress_bar=True), getattr(transformer.auto_model.config, "_commit_hash", None)
    if chunk_size > max_length:
        raise ValueError("chunk_size must not exceed max_length")
    rows = []
    for doc in docs:
        chunks, lengths = chunk_text(doc, transformer.tokenizer, chunk_size, chunk_overlap)
        vectors = encoder.encode(chunks, batch_size=batch_size, convert_to_numpy=True,
                                 show_progress_bar=False)
        weights = np.asarray(lengths, dtype=np.float32)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        rows.append(np.average(vectors, axis=0, weights=weights))
    return np.vstack(rows), getattr(transformer.auto_model.config, "_commit_hash", None)

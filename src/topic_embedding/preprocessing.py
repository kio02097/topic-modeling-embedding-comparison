"""Preprocessing extracted from the supplied research notebook."""
import re

CUSTOM_STOPWORDS = {
    "bbc": ["say", "mr"],
    "20ng": ["say", "_", "maxaxaxaxaxaxaxaxaxaxaxaxaxaxax",
             "mg9vg9vg9vg9vg9vg9vg9vg9vg9vg9vg9vg9vg9vg9vg9v",
             "maxaxaxaxaxaxaxaxaxaxaxaxaxaxaxq"],
    "imdb": ["br", "s"],
}
DATASETS = {"bbc": "SetFit/bbc-news", "20ng": "SetFit/20_newsgroups", "imdb": "SetFit/imdb"}


def clean_text(text, stopwords):
    # Preserve notebook regex and its whitespace-based stopword matching.
    text = re.sub(r"[^A-Za-z0-9\w\s,\.?!]", "", str(text))
    return " ".join(word for word in text.lower().split() if word not in stopwords)


def prepare_documents(dataset, min_tokens=5, limit=None, revision=None):
    import nltk
    import spacy
    from datasets import load_dataset, concatenate_datasets
    from nltk.corpus import stopwords

    try:
        english_stopwords = set(stopwords.words("english"))
    except LookupError as exc:
        raise RuntimeError("Run: python -m nltk.downloader stopwords") from exc
    nlp = spacy.load("en_core_web_sm")
    ds = load_dataset(DATASETS[dataset], revision=revision)
    # This is an unsupervised corpus-level evaluation, not a held-out test.
    corpus = concatenate_datasets([ds["train"], ds["test"]])
    frame = corpus.to_pandas()
    frame.insert(0, "document_id", range(len(frame)))
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    cleaned = [clean_text(text, english_stopwords) for text in frame["text"]]
    frame["lemma_text"] = [" ".join(t.lemma_ for t in doc) for doc in nlp.pipe(cleaned, batch_size=64)]
    frame["token_count"] = frame["lemma_text"].str.split().map(len)
    frame = frame[frame["token_count"] >= min_tokens].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No documents remain after preprocessing.")
    return frame, getattr(corpus, "_fingerprint", None)

"""Independent topic-count fits, matching the notebook's experiment design."""
from .evaluation import evaluate_topics


def evaluate_grid(docs, vectorizer, topic_counts, seed, embeddings=None,
                  umap_seed=None, umap_components=5, lda=False):
    from gensim.corpora import Dictionary

    analyzer = vectorizer.build_analyzer()
    texts = [analyzer(doc) for doc in docs]
    dictionary = Dictionary(texts)
    if not dictionary:
        raise ValueError("Empty vocabulary after vectorization.")
    if not lda:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP
        if len(docs) <= umap_components + 1:
            raise ValueError("Too few documents for the configured UMAP dimension.")
    else:
        from gensim.models import LdaModel
        corpus = [dictionary.doc2bow(text) for text in texts]
    rows, keywords = [], []
    for count in topic_counts:
        if lda:
            model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=count,
                             random_state=seed, passes=10, alpha="auto")
            words = [[w for w, _ in model.show_topic(t, topn=10)] for t in range(count)]
            tids = list(range(count))
            outlier_fraction = None
        else:
            model = BERTopic(embedding_model=None, vectorizer_model=vectorizer,
                             umap_model=UMAP(n_neighbors=15, n_components=umap_components,
                                             min_dist=0.0, metric="cosine", random_state=umap_seed),
                             hdbscan_model=HDBSCAN(min_cluster_size=10, metric="euclidean",
                                                   cluster_selection_method="eom", prediction_data=True),
                             min_topic_size=10, nr_topics=count, top_n_words=10,
                             calculate_probabilities=False)
            assigned, _ = model.fit_transform(docs, embeddings=embeddings)
            topics = model.get_topics()
            tids = sorted(t for t in topics if t != -1)
            words = [[w for w, _ in topics[t] if w] for t in tids]
            outlier_fraction = sum(t == -1 for t in assigned) / len(assigned)
        row = {"nr_topics": count, "n_clusters": len(tids), "outlier_fraction": outlier_fraction}
        row.update(evaluate_topics(words, texts, dictionary))
        rows.append(row)
        keywords.extend({"nr_topics": count, "topic": tid, "words": terms} for tid, terms in zip(tids, words))
    return rows, keywords

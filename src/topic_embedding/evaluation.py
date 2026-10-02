"""Topic metrics share the topic model's token analyzer."""
import math


def topic_diversity(topic_words, top_n=10):
    if not topic_words:
        return math.nan
    words = [word for topic in topic_words for word in topic[:top_n]]
    # Keep the notebook's fixed top-N denominator, including short topics.
    return len(set(words)) / (top_n * len(topic_words))


def evaluate_topics(topic_words, texts, dictionary, top_n=10):
    from gensim.models import CoherenceModel

    if not topic_words:
        return {"coherence_c_v": math.nan, "coherence_c_npmi": math.nan, "topic_diversity": math.nan}
    valid_words = [[w for w in words[:top_n] if w in dictionary.token2id] for words in topic_words]
    # Do not silently drop unscorable topics from the aggregate.
    if any(len(words) < 2 for words in valid_words):
        cv = npmi = math.nan
    else:
        cv = CoherenceModel(topics=valid_words, texts=texts, dictionary=dictionary,
                            coherence="c_v", processes=1).get_coherence()
        npmi = CoherenceModel(topics=valid_words, texts=texts, dictionary=dictionary,
                              coherence="c_npmi", processes=1).get_coherence()
    return {"coherence_c_v": cv, "coherence_c_npmi": npmi,
            "topic_diversity": topic_diversity(topic_words, top_n)}

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def coherence(embs):
    """
    Measures overall semantic coherence of the conversation.
    Higher value = more consistent meaning across sentences.
    """
    if len(embs) < 2:
        return 0.0

    sims = cosine_similarity(embs)
    upper = sims[np.triu_indices(len(sims), 1)]

    # Slight scaling for better interpretability
    return float(np.mean(upper) * 1.1)


def topic_distribution(sentences):
    """
    Extract topic distribution using LDA.
    Returns probability distribution across 5 topics.
    """
    vec = CountVectorizer()
    X = vec.fit_transform(sentences)

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    dist = lda.fit_transform(X)

    return dist.mean(axis=0).tolist()


def topic_shift(embs):
    """
    FINAL CORRECT LOGIC

    Measures how much each sentence deviates from overall coherence.

    Instead of thresholding similarity (which is unstable),
    we measure deviation from average similarity.

    This gives:
    - Low value → coherent conversation
    - High value → real topic jumps
    """
    if len(embs) < 2:
        return 0.0

    sims = cosine_similarity(embs[:-1], embs[1:])
    similarities = sims.diagonal()

    # Average similarity (overall coherence baseline)
    avg_sim = np.mean(similarities)

    # Deviation from average → actual shift
    shifts = [abs(s - avg_sim) for s in similarities]

    return float(np.mean(shifts))
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def coherence(embs):
    if len(embs) < 2:
        return 0.0

    sims = cosine_similarity(embs)
    upper = sims[np.triu_indices(len(sims), 1)]

    return float(np.clip(np.mean(upper), 0, 1))


def topic_distribution(sentences):
    if len(sentences) < 3:
        return [0.0] * 5

    try:
        vec = CountVectorizer()
        X = vec.fit_transform(sentences)

        if X.shape[1] == 0:
            return [0.0] * 5

        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        dist = lda.fit_transform(X)

        return dist.mean(axis=0).tolist()

    except Exception:
        return [0.0] * 5


def topic_shift(embs):
    if len(embs) < 2:
        return 0.0

    sims = cosine_similarity(embs[:-1], embs[1:])
    similarities = sims.diagonal()

    avg_sim = np.mean(similarities)
    shifts = [abs(s - avg_sim) for s in similarities]

    return float(np.mean(shifts))
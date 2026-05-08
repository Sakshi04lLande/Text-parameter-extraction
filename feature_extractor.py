# feature_extractor.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentiment_emotion import extract_emotion_features
from discourse import coherence, topic_distribution, topic_shift

def MATTR(tokens, w=30):
    if len(tokens) < w:
        return len(set(tokens)) / max(len(tokens), 1)
    return np.mean([
        len(set(tokens[i:i+w])) / w
        for i in range(len(tokens)-w+1)
    ])

def extract(text, nlp, embs, psych):
    try:
        tokens = nlp.get("tokens", [])
        sentences = nlp.get("sentences", [])
    except Exception:
        return {}

    # ---------------- LEXICAL ----------------
    total = len(tokens)
    unique = len(set(tokens))
    hapax = len([t for t in set(tokens) if tokens.count(t) == 1])
    sentence_count = len(sentences)
    repetition_ratio = 1 - (unique / max(total, 1))

    # ---------------- EMOTION ----------------
    emotion = extract_emotion_features(sentences)

    # sentence-level sentiment for dynamics
    per_sent = [
        extract_emotion_features([s])["Overall Sentiment Score"]
        for s in sentences
    ]
    sentiment_variance = float(np.var(per_sent)) if per_sent else 0.0

    sentiment_slope = (
        float(np.polyfit(range(len(per_sent)), per_sent, 1)[0])
        if len(per_sent) > 1 else 0.0
    )

    emotional_volatility_score = abs(sentiment_slope)

    # ---------------- DISCOURSE ----------------
    sem_coh = coherence(embs)
    topics = topic_distribution(sentences)
    tshift = topic_shift(embs)

    sim_matrix = cosine_similarity(embs) if len(embs) > 0 else np.array([[0]])
    if len(embs) > 1:
        upper_vals = sim_matrix[np.triu_indices(len(sim_matrix), 1)]
        max_sim = float(np.max(upper_vals))if len(upper_vals) > 0 else 0.0
    else:
        max_sim = 0.0
    if len(embs) > 1:
        first_last_sim = float(cosine_similarity([embs[0]], [embs[-1]])[0][0])
    else:
        first_last_sim = 0.0
    max_sim = max(min(max_sim, 1.0), -1.0)
    first_last_sim = max(min(first_last_sim, 1.0), -1.0)

    # ---------------- PSYCHOLOGY ----------------
    def avg(k):
        try:
            values = [d.get(k, 0) for d in psych if isinstance(d, dict)]
            return float(np.mean(values)) if values else 0.0
        except Exception:
            return 0.0
    past = np.mean([d["time_focus"] == "past" for d in psych])
    present = np.mean([d["time_focus"] == "present" for d in psych])
    future = np.mean([d["time_focus"] == "future" for d in psych])

    # ---------------- FINAL 49 PARAMETERS ----------------
    return {
        # 1–8 Lexical
        "total_word_count": total,
        "unique_word_count": unique,
        "type_token_ratio": unique / max(total, 1),
        "moving_average_ttr": MATTR(tokens),
        "hapax_legomena_ratio": hapax / max(total, 1),
        "sentence_count": sentence_count,
        "average_sentence_length": nlp["avg_sentence_length"],
        "repetition_ratio": repetition_ratio,

        # 9–15 POS
        "noun_ratio": nlp["noun_ratio"],
        "verb_ratio": nlp["verb_ratio"],
        "adjective_ratio": nlp["adj_ratio"],
        "adverb_ratio": nlp["adv_ratio"],
        "pronoun_ratio": nlp["pronoun_ratio"],
        "modal_verb_ratio": nlp["modal_ratio"],
        "negation_ratio": nlp["negation_ratio"],

        # 16–19 Syntax
        "parse_tree_depth": nlp["parse_tree_depth"],
        "avg_dependency_length": nlp["avg_dep_length"],
        "clause_count": nlp["clause_count"],
        "subordinate_clause_ratio": nlp["subordinate_clause_ratio"],

        # 20–34 Emotion (ALL 15 ✅)
        "Positive Emotion Word Ratio": emotion["Positive Emotion Word Ratio"],
        "Negative Emotion Word Ratio": emotion["Negative Emotion Word Ratio"],
        "Overall Sentiment Score": emotion["Overall Sentiment Score"],

        "Fear Frequency": emotion["Fear Frequency"],
        "Sadness Frequency": emotion["Sadness Frequency"],
        "Anger Frequency": emotion["Anger Frequency"],
        "Joy Frequency": emotion["Joy Frequency"],
        "Disgust Frequency": emotion["Disgust Frequency"],
        "Surprise Frequency": emotion["Surprise Frequency"],

        "Emotional Intensity Ratio": emotion["Emotional Intensity Ratio"],
        "Max Negative Emotion": emotion["Max Negative Emotion"],
        "Negative Emotion Spike Count": emotion["Negative Emotion Spike Count"],

        "sentiment_variance": sentiment_variance,
        "sentiment_trend_slope": sentiment_slope,
        "emotional_volatility_score": emotional_volatility_score,

        # 35–40 Semantics
        "semantic_coherence_score": sem_coh,
        "topic_distribution_vector": topics,
        "topic_shift_frequency": tshift,
        "sentence_embedding_vector": embs.mean(axis=0)[:10].tolist(),
        "max_sentence_similarity": max_sim,
        "first_last_sentence_similarity": first_last_sim,

        # 41–49 Psychology & Cognitive
        "absolutist_score": avg("absolutist_score"),
        "helplessness_score": avg("helplessness_score"),
        "catastrophizing_score": avg("catastrophizing_score"),
        "external_locus_score": avg("external_locus_score"),
        "rumination_score": avg("rumination_score"),
        "uncertainty_score": avg("uncertainty_score"),
        "avoidance_score": avg("avoidance_score"),
        "threat_score": avg("threat_score"),
        "self_reference_density": nlp["self_reference_density"],
        "past_focus_ratio": past,
        "present_focus_ratio": present,
        "future_focus_ratio": future,
        "filler_ratio": nlp["filler_ratio"],
        "cognitive_load_score": nlp["cognitive_load_score"]
    }
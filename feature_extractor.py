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
        "first_person_singular_pronoun_frequency": nlp["pronoun_ratio"],
        "modal_verb_frequency": nlp["modal_ratio"],
        "negative_frequency": nlp["negation_ratio"],

        # 16–19 Syntax
        "parse_tree_depth": nlp["parse_tree_depth"],
        "avg_dependency_length": nlp["avg_dep_length"],
        "clause_count": nlp["clause_count"],
        "subordinate_clause_ratio": nlp["subordinate_clause_ratio"],

        # 20–34 Emotion (ALL 15 ✅)
        "positive_emotion_word_ratio": emotion["Positive Emotion Word Ratio"],
        "negative_emotion_word_ratio": emotion["Negative Emotion Word Ratio"],
        "overall_sentiment_score": emotion["Overall Sentiment Score"],

        "fear_word_frequency": emotion["Fear Frequency"],
        "sadness_word_frequency": emotion["Sadness Frequency"],
        "anger_word_frequency": emotion["Anger Frequency"],
        "joy_frequency": emotion["Joy Frequency"],
        "disgust_frequency": emotion["Disgust Frequency"],
        "surprise_frequency": emotion["Surprise Frequency"],

        "emotional_intensity_ratio": emotion["Emotional Intensity Ratio"],
        "max_negative_emotion": emotion["Max Negative Emotion"],
        "negative_emotion_spike_count": emotion["Negative Emotion Spike Count"],

        "sentiment_variance": sentiment_variance,
        "sentiment_trajectory_slope": sentiment_slope,
        "emotional_volatility_score": emotional_volatility_score,

        # 35–40 Semantics
        "semantic_coherence_score": sem_coh,
        "topic_distribution_vector": topics,
        "sentence_embedding_vector": (
             np.mean(embs, axis=0)[:10].tolist()
             if len(embs) > 0 else [0.0] * 10
       ),
        "max_sentence_similarity": max_sim,
        "first_last_sentence_similarity": first_last_sim,

        # 41–49 Psychology & Cognitive
        "absolutist_word_frequency": avg("absolutist_score"),
        "helplessness_phrase_frequency": avg("helplessness_score"),
        "catastrophizing_indicators": avg("catastrophizing_score"),
        "external_locus_of_control_score": avg("external_locus_score"),
        "rumination_phrase_frequency": avg("rumination_score"),
        "uncertainty_word_frequency": avg("uncertainty_score"),
        "avoidance_language_frequency": avg("avoidance_score"),
        "threat_anticipation_language": avg("threat_score"),
        "self_reference_density": nlp["self_reference_density"],
        "past_focused_word_ratio": past,
        "present_focused_word_ratio": present,
        "future_focused_word_ratio": future,
        "filler_word_frequency": nlp["filler_ratio"],
        "cognitive_load_score": nlp["cognitive_load_score"]
    }
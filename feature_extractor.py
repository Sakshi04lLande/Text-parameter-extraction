import numpy as np
from sentiment_emotion import extract_emotion_features
from discourse import coherence, topic_distribution, topic_shift

def MATTR(tokens, window=30):
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens)

    scores = []
    for i in range(len(tokens) - window + 1):
        chunk = tokens[i:i+window]
        scores.append(len(set(chunk)) / len(chunk))

    return float(np.mean(scores))


def extract(text, nlp, embs, llm):
    tokens = nlp["tokens"]
    sentences = nlp["sentences"]
    total = len(tokens)

    unique = len(set(tokens))
    hapax = len([w for w in set(tokens) if tokens.count(w) == 1])

    emotion = extract_emotion_features(sentences)

    per_sent = [extract_emotion_features([s])["overall_sentiment_score"] for s in sentences]
    sent_var = float(np.var(per_sent)) if per_sent else 0

    coh = coherence(embs)
    topics = topic_distribution(sentences)
    t_shift = topic_shift(embs)

    # LLM features
    absolutist = np.mean([d["absolutist"] for d in llm])
    helpless = np.mean([d["helplessness"] for d in llm])
    cat = np.mean([d["catastrophizing"] for d in llm])
    locus = np.mean([d["external_locus"] for d in llm])
    rumi = np.mean([d["rumination"] for d in llm])
    unct = np.mean([d["uncertainty"] for d in llm])
    avoid = np.mean([d["avoidance"] for d in llm])
    threat = np.mean([d["threat_anticipation"] for d in llm])
    selfref = np.mean([d["self_reference_density"] for d in llm])

    # ✅ self-reference fallback
    fp = ["i","me","my","mine","myself"]
    rule_self = sum(1 for w in tokens if w in fp) / total if total else 0
    selfref = max(selfref, rule_self)

    # time focus
    past = np.mean([d["time_focus"] == "past" for d in llm])
    present = np.mean([d["time_focus"] == "present" for d in llm])
    future = np.mean([d["time_focus"] == "future" for d in llm])

    # fallback
    if past == 0 and future == 0:
        past_words = ["was","were","had","before","yesterday"]
        future_words = ["will","future","tomorrow","might"]

        past = sum(1 for w in tokens if w in past_words) / total
        future = sum(1 for w in tokens if w in future_words) / total
        present = 1 - (past + future)

    return {
        "total_word_count": total,
        "unique_word_count": unique,
        "type_token_ratio": unique / total if total else 0,
        "moving_average_ttr": MATTR(tokens),
        "hapax_legomena_ratio": hapax / total if total else 0,

        "positive_emotion_ratio": emotion["positive_emotion_ratio"],
        "negative_emotion_ratio": emotion["negative_emotion_ratio"],
        "overall_sentiment_score": emotion["overall_sentiment_score"],
        "fear_frequency": emotion["fear_frequency"],
        "sadness_frequency": emotion["sadness_frequency"],
        "anger_frequency": emotion["anger_frequency"],
        "sentiment_variance": sent_var,

        "semantic_coherence_score": coh,
        "sentence_embedding_vector": embs.mean(axis=0).tolist()[:10],
        "topic_distribution_vector": topics,
        "topic_shift_frequency": t_shift,

        "noun_ratio": nlp["noun_ratio"],
        "verb_ratio": nlp["verb_ratio"],
        "adjective_ratio": nlp["adj_ratio"],
        "adverb_ratio": nlp["adv_ratio"],

        "average_sentence_length": nlp["avg_sentence_length"],
        "parse_tree_depth": nlp["parse_tree_depth"],

        "absolutist_thinking_frequency": absolutist,
        "helplessness_frequency": helpless,
        "catastrophizing_score": cat,
        "external_locus_of_control": locus,
        "rumination_frequency": rumi,
        "uncertainty_frequency": unct,
        "avoidance_language_frequency": avoid,
        "threat_anticipation_frequency": threat,
        "self_reference_density": selfref,

        "past_focus_ratio": past,
        "present_focus_ratio": present,
        "future_focus_ratio": future,

        "response_length_per_prompt": nlp["avg_sentence_length"]
    }
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

LABELS = ["negative", "neutral", "positive"]


def get_sentiment_scores(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return probs


def extract_emotion_features(sentences):
    all_scores = np.array([get_sentiment_scores(s) for s in sentences])

    # Balanced aggregation
    scores = all_scores.mean(axis=0) * 0.7 + all_scores.max(axis=0) * 0.3

    neg = float(scores[0])
    neu = float(scores[1])
    pos = float(scores[2])

    # Normalize
    total = neg + pos + 1e-6
    pos_ratio = pos / total
    neg_ratio = neg / total

    return {
        "positive_emotion_ratio": pos_ratio,
        "negative_emotion_ratio": neg_ratio,
        "overall_sentiment_score": pos_ratio - neg_ratio,

        # Approximate mapping (since model is sentiment, not emotion-specific)
        "fear_frequency": neg * 0.3,
        "sadness_frequency": neg * 0.5,
        "anger_frequency": neg * 0.2
    }
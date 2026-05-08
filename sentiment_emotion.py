# sentiment_emotion.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "bhadresh-savani/bert-base-go-emotion"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

EMOTION_INDEX = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "fear": 3,
    "disgust": 4,
    "surprise": 5
}

def extract_emotion_features(sentences):
    if not sentences:
        return {k: 0.0 for k in [
            "positive_emotion_ratio",
            "negative_emotion_ratio",
            "overall_sentiment_score",
            "fear_frequency",
            "sadness_frequency",
            "anger_frequency",
            "joy_frequency",
            "disgust_frequency",
            "surprise_frequency",
            "emotional_intensity_ratio",
            "max_negative_emotion",
            "negative_emotion_spike_count"
        ]}

    tokens = tokenizer(sentences, return_tensors="pt",padding=True, truncation=True,max_length=512)
    with torch.no_grad():
                 logits = model(**tokens).logits

    probs = torch.softmax(logits, dim=1).cpu().numpy()
            
    emotion_vectors = probs

    
    joy = emotion_vectors[:, EMOTION_INDEX["joy"]]
    sadness = emotion_vectors[:, EMOTION_INDEX["sadness"]]
    anger = emotion_vectors[:, EMOTION_INDEX["anger"]]
    fear = emotion_vectors[:, EMOTION_INDEX["fear"]]
    disgust = emotion_vectors[:, EMOTION_INDEX["disgust"]]
    surprise = emotion_vectors[:, EMOTION_INDEX["surprise"]]

    positive = joy
    negative = sadness + anger + fear + disgust

    overall = positive.mean() - negative.mean()

    # dynamics
    spike_count = int((negative > negative.mean() + negative.std()).sum())

    return {
        "Positive Emotion Word Ratio": float(positive.mean()),
        "Negative Emotion Word Ratio": float(negative.mean()),
        "Overall Sentiment Score": float(overall),

        "Fear Frequency": float(fear.mean()),
        "Sadness Frequency": float(sadness.mean()),
        "Anger Frequency": float(anger.mean()),
        "Joy Frequency": float(joy.mean()),
        "Disgust Frequency": float(disgust.mean()),
        "Surprise Frequency": float(surprise.mean()),

        "Emotional Intensity Ratio": float(np.mean(np.max(emotion_vectors, axis=1))),
        "Max Negative Emotion": float(negative.max()),
        "Negative Emotion Spike Count": float(spike_count)
    }

# Mental Health NLP Analyzer API

A production-ready FastAPI-based API that analyzes multilingual conversations and extracts linguistic, emotional, discourse, and psychological features using NLP and Deep Learning models.

---

# Features

* Multilingual Support:

  * English
  * Hindi
  * Marathi

* NLP Feature Extraction

* Emotion Detection

* Psychological Pattern Analysis

* Semantic Coherence Analysis

* Topic Shift Detection

* 49 Structured Mental Health Features

---

# API Endpoint

## POST `/analyze`

Analyzes client conversation text and returns structured psychological and linguistic features.

---

# Request Format

```json
{
  "text": "Client: I feel anxious and worried these days."
}
```

---

# Response Format

```json
{
  "status": "success",
  "features": {
    "total_word_count": 120,
    "unique_word_count": 80,
    "Overall Sentiment Score": -0.42
  }
}
```

---

# How to Run

## Local Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Stanza Models

```bash
python stanza_download.py
```

### 3. Run API

```bash
uvicorn app:app --host 0.0.0.0 --port 8025
```

API will start at:

```text
http://localhost:8025
```

Swagger Documentation:

```text
http://localhost:8025/docs
```

---

# Docker Setup

```bash
docker compose up --build
```

---

# How to Use

Example API request using Python requests library:

```python
import requests
import json

url = "http://localhost:8025/analyze"

payload = {
    "text": "Client: I feel very stressed and anxious these days."
}

headers = {
    "x-api-key": "your_secure_api_key"
}

response = requests.post(
    url,
    json=payload,
    headers=headers
)

print(response.json())
```

---

# Required Files

## payload.json

Contains sample API input payload.

Example:

```json
{
  "text": "Client: I feel anxious and overthink everything."
}
```

---

## output.json

Contains sample API response.

---

## run.py

Ready-to-run script that:

* Reads payload.json
* Calls API
* Prints API response

Run using:

```bash
python run.py
```

---

# Environment Variables

Create a `.env` file:

```env
MENTAL_HEALTH_ANALYZER_KEY=your_secure_generated_api_key
```

---

# example.env

```env
MENTAL_HEALTH_ANALYZER_KEY=replace_with_your_api_key
```

Do NOT commit actual `.env` files.

---

# Project Structure

```text
Mental_health/
│
├── app.py
├── run.py
├── payload.json
├── output.json
├── README.md
├── requirements.txt
├── example.env
├── .env
├── .gitignore
├── Dockerfile
├── docker-compose.yml
│── main.py
├── feature_extractor.py
├── discourse.py
├── psychology.py
├── sentiment_emotion.py
├── embedder.py
├── nlp_engine.py
├── stanza_download.py
└── emotion_model_download.py
```

---

# Tech Stack

* FastAPI
* Stanza NLP
* HuggingFace Transformers
* Sentence Transformers
* Scikit-learn
* NumPy
* PyTorch

---

# Security

* API authentication enabled using API keys
* Environment variables stored using `.env`
* Do not expose secret keys publicly

---

# Production Notes

* Recommended RAM: 4GB–8GB
* GPU optional
* First startup may take time because NLP models are downloaded

---

# Use Cases

* Mental Health Analytics
* AI Therapy Assistants
* Behavioral Research
* Emotion-Aware Chatbots

---



Here is a clean, professional **README.md** tailored to your project. It reflects what your code actually does, not just generic documentation.

---

# 🧠 Psychological Analysis API

A FastAPI-based system that analyzes human conversations to extract **linguistic, emotional, and psychological features** using NLP, embeddings, and LLM-based reasoning.

---

## 🚀 Overview

This project processes conversational text (especially therapist–client conversations) and extracts deep psychological insights such as:

* Sentiment & emotional patterns
* Cognitive distortions (e.g., catastrophizing, helplessness)
* Linguistic complexity
* Topic coherence & shifts
* Self-focus and time orientation

It is designed for **mental health analysis, research, and AI-driven behavioral insights**.

---

## ⚙️ Key Features

### 1. 🗣 Conversation Processing

* Accepts full conversation input
* Automatically extracts **Client-only text**
* Handles multi-line and noisy input formats 

---

### 2. 🌍 Multilingual NLP Engine

* Supports:

  * English
  * Hindi
  * Marathi
* Uses **Stanza** for:

  * Tokenization
  * POS tagging
  * Dependency parsing 

---

### 3. 📊 Feature Extraction (49+ Parameters)

#### 🔤 Linguistic Features

* Word count, vocabulary richness (TTR, MATTR)
* Sentence length & structure
* POS ratios (noun, verb, adjective, adverb)

#### 😊 Emotional Features

* Positive / Negative sentiment
* Fear, sadness, anger estimation
* Sentiment variance 

#### 🧠 Psychological Signals (LLM-based)

* Absolutist thinking
* Helplessness
* Catastrophizing
* Rumination
* Avoidance behavior
* Threat anticipation
* External locus of control
* Self-reference density 

#### 🧩 Discourse Features

* Semantic coherence
* Topic distribution (LDA)
* Topic shift detection 

#### ⏳ Time Orientation

* Past / Present / Future focus

---

### 4. 🤖 Embeddings

* Uses **Sentence Transformers**
* Multilingual semantic embeddings for deep analysis 

---

### 5. ⏱ Latency Tracking

Tracks execution time for:

* NLP processing
* Embeddings
* LLM inference
* Feature extraction 

---

### 6. 🔐 Secure API

* API key authentication
* Input size validation
* Error-safe responses 

---

## 🧱 Project Structure

```
├── api.py                  # FastAPI endpoints
├── main.py                 # Pipeline runner with latency tracking
├── nlp_engine.py           # Language detection + NLP processing
├── feature_extractor.py    # Core feature extraction (49+ features)
├── sentiment_emotion.py    # Sentiment analysis
├── discourse.py            # Coherence & topic modeling
├── embedder.py             # Sentence embeddings
├── psychology.py           # LLM-based psychological analysis
├── indicbert_download.py   # Model download (optional)
├── stanza_download.py      # Language model setup
├── requirements.txt        # Dependencies
```

---

## 📦 Installation

```bash
git clone <your-repo-url>
cd project-folder

pip install -r requirements.txt
```

---

## ⬇️ Download Models

Run:

```bash
python stanza_download.py
```

---

## 🔑 Environment Setup

Create a `.env` file:

```
API_KEY=your_secret_key
ANTHROPIC_API_KEY=your_anthropic_key
```

---

## ▶️ Run the API

```bash
uvicorn api:app --reload
```

---

## 📡 API Usage

### Endpoint:

```
POST /analyze
```

### Headers:

```
x-api-key: your_secret_key
```

### Request Body:

```json
{
  "conversation": "Assistant: Hello...\nClient: I feel very sad..."
}
```

---

### ✅ Response Example

```json
{
  "status": "success",
  "client_text": "I feel very sad...",
  "analysis": {
    "total_word_count": 120,
    "positive_emotion_ratio": 0.3,
    "negative_emotion_ratio": 0.7,
    "semantic_coherence_score": 0.82,
    "catastrophizing_score": 0.4,
    ...
  },
  "latency": {
    "nlp_time": 0.45,
    "embedding_time": 0.32,
    "llm_total_time": 1.2,
    "total_time": 2.1
  }
}
```

---

## 🧪 Running via CLI

```bash
python main.py
```

Paste conversation input and get:

* Extracted client text
* Feature analysis
* Latency breakdown

---

## 📌 Input Format Requirement

Conversation must include:

```
Client: ...
Assistant: ...
```

Only **Client text** is analyzed.

---

## 🛠 Technologies Used

* FastAPI
* Stanza NLP
* Sentence Transformers
* HuggingFace Transformers
* Scikit-learn (LDA)
* Anthropic Claude (LLM reasoning)
* NumPy

---

## 🎯 Use Cases

* Mental health analysis systems
* AI therapy assistants
* Behavioral research
* Emotion-aware chatbots
* Psychological profiling tools

---

## ⚠️ Limitations

* Requires properly formatted conversation input




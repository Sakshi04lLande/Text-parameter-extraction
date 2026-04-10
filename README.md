#  Psychological Analysis API

A FastAPI-based system that analyzes conversations and extracts **linguistic, emotional, and psychological insights** using NLP, embeddings, and LLM-based reasoning.

---

#  Quick Start (Recommended – Docker)

If you just want to run the project quickly:

```bash
git clone <your-repo-url>
cd project-folder

docker compose up -d --build
```

Then open:

```
http://localhost:8025/docs
```

---

#  Overview

This system analyzes **Client–Assistant conversations** and extracts deep insights such as:

* Emotional patterns (positive/negative sentiment)
* Cognitive distortions (catastrophizing, helplessness)
* Linguistic complexity
* Topic coherence and shifts
* Self-focus and time orientation

It is designed for:

* Mental health analysis
* AI therapy systems
* Behavioral research
* Emotion-aware applications

---

#  How It Works (Simple Flow)

1. Extract **Client text** from conversation
2. Run **NLP processing (Stanza)**
3. Generate **sentence embeddings**
4. Analyze psychology using **LLM (Anthropic)**
5. Compute **49+ features**
6. Return structured JSON output

---

#  Installation (Manual Setup)

```bash
git clone <your-repo-url>
cd project-folder

pip install -r requirements.txt
```

---

#  Download Required Models

```bash
python stanza_download.py
```

---

#  Environment Setup

Create a `.env` file:

```
API_KEY=your_secret_key
ANTHROPIC_API_KEY=your_anthropic_key
```

---

#  Run API (without Docker)

```bash
uvicorn api:app --reload
```

Open:

```
http://localhost:8025/docs
```

---

#  Docker Setup (Recommended)

Build and run:

```bash
docker compose up -d --build
```

Check running containers:

```bash
docker ps
```

---

#  API Usage

## Endpoint

```
POST /analyze
```

## Headers

```
x-api-key: your_secret_key
```

## Request Body

```json
{
  "conversation": "Assistant: Hello\nClient: I feel very stressed"
}
```

---

##  Response Example

```json
{
  "status": "success",
  "client_text": "I feel very stressed",
  "analysis": {
    "total_word_count": 120,
    "semantic_coherence_score": 0.82,
    "catastrophizing_score": 0.4
  },
  "latency": {
    "total_time": 2.1
  }
}
```

---

#  Input Format Requirement


To help you understand the expected input format, an example file is included:

```text
demo.txt
```

This file contains sample conversations in the correct format (single-line, labeled with `Client:` and `Assistant:`).

---

##  How to Use

1. Open `demo.txt`
2. Copy any conversation (English / Marathi / Hindi)
3. Send it as input to the API

---

##  Input Format Rules

* Conversation must be in **single line**
* Must include labels:

  ```text
  Client: ...
  Assistant: ...
  ```
* Do not mix multiple languages in one input
* Only **Client text** will be analyzed internally

---

##  Example API Request

```json
{
  "conversation": "Client: I feel stressed and anxious Assistant: Tell me more about that feeling"
}
```

---

This file is especially useful for:

* First-time users
* Testing API quickly
* Understanding correct formatting


---

#  Test API Quickly

Open Swagger UI:

```
http://localhost:8025/docs
```

---

#  Project Structure

```
├── api.py
├── main.py
├── nlp_engine.py
├── feature_extractor.py
├── sentiment_emotion.py
├── discourse.py
├── embedder.py
├── psychology.py
├── stanza_download.py
├── requirements.txt
```

---

# Technologies Used

* FastAPI
* Stanza NLP
* Sentence Transformers
* HuggingFace Transformers
* Scikit-learn
* Anthropic Claude (LLM)
* NumPy

---

#  Limitations

* Requires properly formatted input
* Heavy models → needs good RAM (6GB+ recommended)
* LLM calls depend on API key

---

#  Use Cases

* Mental health analytics
* AI therapy assistants
* Behavioral research tools
* Emotion-aware chatbots

---

#  Security Note

Never expose your `.env` file publicly.

---

# Summary

This project combines:

* NLP
* Deep learning
* LLM reasoning

to generate structured psychological insights from conversations.

---

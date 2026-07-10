# 📰 Fake News Detection System

**An intelligent web application that classifies news articles as Real or Fake using NLP and deep learning, backed by a confidence score for every prediction.**

Combines a MERN stack frontend/backend with a Python-based AI microservice, analyzing linguistic patterns, writing style, and semantic meaning to flag likely misinformation in real time.

---

## 🚀 Overview

As misinformation spreads faster than it can be fact-checked, this project provides a practical first line of defense: paste in a news article, and get an instant classification — Real or Fake — with a confidence score attached, so the result can be weighed rather than blindly trusted.

The system separates concerns cleanly: a **Node.js/Express API** handles routing and client communication, while a dedicated **Python/Flask microservice** runs the actual transformer-based inference — keeping the ML workload isolated from the main application layer.

---

## 🎯 Key Features

### 🧠 AI-Powered Classification
- Classifies news content as **Real** or **Fake**
- Returns a confidence score (%) alongside every prediction
- Powered by deep learning models trained on real-world labeled datasets

### 🔍 Advanced Text Analysis
Analyzes text beyond surface-level keywords:
- Linguistic patterns and writing tone
- Structural and stylistic cues common in fabricated content
- Semantic meaning, not just word matching

### ⚡ Real-Time Predictions
- Fast inference via a dedicated Flask microservice
- Seamless request handling through the MERN backend gateway

### 🌐 Full-Stack Web Application
- Clean, responsive React interface
- Paste an article, get an instant result — no unnecessary friction

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React.js |
| **Backend (API Gateway)** | Node.js, Express.js |
| **AI Microservice** | Python, Flask |
| **ML/NLP** | PyTorch, Hugging Face Transformers |
| **Model Architecture** | Transformer-based (BERT-like), fine-tuned for classification |

---

## 📚 Dataset

- Sourced from **Hugging Face Datasets** (labeled Fake News collections)
- Preprocessing pipeline includes:
  - Text cleaning and normalization
  - Tokenization
  - Stopword removal
  - Encoding via transformer tokenizers

---

## 🧩 System Architecture

.## 🔄 Workflow

1. User submits a news article through the React UI
2. Request is routed to the Node.js/Express backend
3. Backend forwards the text to the Flask AI microservice
4. The transformer model runs inference on the input
5. Model returns a prediction and confidence score
6. Result is rendered back to the user in real time

---

## 🔬 Model Details

- Transformer-based architecture (BERT-like)
- Fine-tuned specifically on fake news classification datasets
- Optimized for:
  - **Accuracy** — correctly distinguishing real vs. fabricated content
  - **Generalization** — performing reliably across topics and writing styles it wasn't explicitly trained on
  - **Low latency** — fast enough for real-time, interactive use

---

## 📊 Example Output

**Input:**
> "Breaking news: Scientists confirm water is dry!"

**Result:**
```json
{
  "prediction": "Fake News ❌",
  "confidence": "98.7%"
}
```

---

## 🔒 Security & Reliability

- Input validation to guard against malicious or malformed payloads
- Microservice architecture allows independent scaling of the AI layer
- API rate limiting *(planned enhancement)*

---

## 🌟 Use Cases

- **Media organizations** — pre-screening content before publication
- **Social platforms** — filtering or flagging likely misinformation
- **Educational tools** — teaching media literacy and critical thinking
- **General users** — a quick sanity check on questionable articles

---

## 📈 Roadmap

- [ ] Multi-language fake news detection
- [ ] Browser extension for live, in-page news verification
- [ ] Source credibility scoring (beyond just article text)
- [ ] Explainable AI — surface *why* an article was flagged as fake
- [ ] Direct integration with social media platforms

---

## 🛠️ Getting Started

### AI Microservice (Flask)
```bash
cd ai-service
pip install -r requirements.txt
python app.py
```

### Backend (Node.js/Express)
```bash
cd backend
npm install
npm start
```

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
```

> Configure the Flask service URL and any model paths in a `.env` file in the backend before running.

---

## ⚠️ Disclaimer

This tool provides a probabilistic classification based on linguistic patterns, not a definitive fact-check. Results should be used as a supporting signal alongside source verification and human judgment, not as a sole source of truth.

---

## 📄 License

MIT

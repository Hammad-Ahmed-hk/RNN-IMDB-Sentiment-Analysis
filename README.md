# 🎬 CineRead — IMDB Sentiment Analysis

> A Streamlit web app that classifies IMDB movie reviews as positive or negative using a PyTorch `nn.RNN` layer on TF-IDF features.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-87.62%25-success?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-IMDB%2050K-yellow?style=flat-square)

---

## 📌 Overview

**CineRead** leverages natural language processing (NLP) techniques to classify user-provided movie reviews as **Positive** or **Negative** with a high degree of confidence. Built with a cinematic UI and a robust deep learning pipeline, it bridges the gap between academic ML models and production-ready web applications.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🎨 **Cinematic UI** | Warm, movie-themed interface built with Streamlit and custom CSS |
| ⚡ **Real-time Analysis** | Instant sentiment classification with probability breakdowns |
| 🧹 **Robust NLP Pipeline** | Advanced text cleaning, stemming, and stop-word removal |
| 📈 **High Accuracy** | **87.62%** test accuracy on the IMDB 50K dataset |

---

## 🧠 Technical Architecture

### The Model

A small PyTorch network (`SentimentRNN` in the notebook):

```
Input: 5,000 TF-IDF features, fed as a sequence of length 1
        ↓
nn.RNN (128 hidden units)
        ↓
Linear layer → Sigmoid → probability 0–1
        ↓
Verdict: Positive / Negative
```

> **Honest note on the architecture.** The whole review is turned into a single TF-IDF vector and given to the RNN as *one* time step. An RNN's strength is reading a sequence of words in order, and that does not happen here, so in practice this behaves like a one-hidden-layer feed-forward network on a bag-of-words. It is a perfectly valid sentiment baseline, but the "RNN" label overstates what it does. A true sequence model would tokenize each review into word indices, use an embedding layer, and feed those to an LSTM or GRU.

### The Pipeline

```
Raw Review Text
      │
      ▼
① Preprocessing   →  Lowercase, strip HTML/URLs, Porter Stemmer
      │
      ▼
② Vectorization   →  TF-IDF (Term Frequency-Inverse Document Frequency)
      │
      ▼
③ Inference       →  Model scores the TF-IDF vector → Final Verdict
```

---

## 🛠️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Hammad-Ahmed-hk/RNN-IMDB-Sentiment-Analysis.git
cd RNN-IMDB-Sentiment-Analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

> Make sure `rnn_model.pth` and `tfidf.pkl` are present in the root directory.

```bash
streamlit run app.py
```

---

## 📊 Dataset & Training

| Parameter | Value |
|---|---|
| **Dataset** | [IMDB 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (`IMDB Dataset.csv`, about 63 MB, included) |
| **Training Samples** | 40,000 |
| **Testing Samples** | 10,000 |
| **Optimizer** | Adam (`lr = 0.001`) |
| **Loss Function** | Binary Cross Entropy (`BCELoss`) |
| **Test Accuracy** | **87.62%** (reported by the author; the notebook has no saved output for this cell, so re-run it to confirm) |

---

## 📁 Project Structure

```
RNN-IMDB-Sentiment-Analysis/
│
├── app.py                        # Streamlit web application
├── RNN_Sentiment_Training.ipynb  # Model training notebook
├── rnn_model.pth                 # Trained RNN model weights
├── tfidf.pkl                     # Saved TF-IDF vectorizer
└── IMDB Dataset.csv              # Raw dataset
```

---

## 👨‍💻 Author

**Hammad Ahmed**
BS Software Engineering Student

[![LinkedIn]="[hammad-ahmed-1a689735b](https://www.linkedin.com/in/hammad-ahmed-1a689735b)"
[![Portfolio]="https://hammad-ahmed-hk.github.io/My-Portfolio/"

---

## 🏷️ Tags

`NLP` `PyTorch` `Streamlit` `Deep-Learning` `RNN` `Sentiment-Analysis` `IMDB` `Python` `TF-IDF`

---

<p align="center">Made with ❤️ and a love for cinema</p>

# 🎬 CineRead — IMDB Sentiment Analysis

> A professional-grade web application that analyzes movie review sentiments using a Recurrent Neural Network (RNN).

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

The core is a **Many-to-One RNN** implemented in PyTorch:

```
Input (5,000 TF-IDF features)
        ↓
Hidden Layer (128 RNN units)
        ↓
Output Layer (Sigmoid activation → probability score 0–1)
        ↓
Verdict: Positive / Negative
```

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
③ Inference       →  RNN processes tensor → Final Verdict
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
pip install streamlit torch scikit-learn nltk numpy pandas
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
| **Dataset** | IMDB 50K Movie Reviews |
| **Training Samples** | 40,000 |
| **Testing Samples** | 10,000 |
| **Optimizer** | Adam (`lr = 0.001`) |
| **Loss Function** | Binary Cross Entropy (`BCELoss`) |
| **Test Accuracy** | **87.62%** |

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

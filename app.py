"""
╔══════════════════════════════════════════════════════════╗
║    🎬 IMDB Sentiment Analysis — Streamlit Web App        ║
║    Model: RNN (PyTorch) + TF-IDF Vectorizer              ║
╚══════════════════════════════════════════════════════════╝

HOW TO RUN:
  1. Place rnn_model.pth and tfidf.pkl in the SAME folder as this file
  2. pip install streamlit torch scikit-learn nltk numpy
  3. streamlit run app.py
"""

import streamlit as st
import pickle
import re
import numpy as np
import torch
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineRead — Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# WARM CINEMATIC PALETTE
#
#   Page background   #FDF6EC  warm cream
#   Surface / card    #FFFFFF  white
#   Deep red          #B5192B  Rotten Tomatoes red
#   Amber accent      #E8881A  warm amber
#   Ink text          #1A1208  near-black warm
#   Body text         #3D2B1F  warm dark brown
#   Muted text        #8A6F5E  warm taupe
#   Border            #E2D5C3  warm sand
#   Positive green    #1A6B35  forest green
#   Section tint      #FFF8F0  warm off-white
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Inter:wght@400;500;600&display=swap');

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background-color: #FDF6EC !important;
    color: #1A1208 !important;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
section[data-testid="stSidebar"] { background: #FFF8F0 !important; }

/* ── Fonts ── */
html, body, * { font-family: 'Inter', sans-serif !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #FDF6EC; }
::-webkit-scrollbar-thumb { background: #E8881A; border-radius: 3px; }

/* ─────────────────────────── HERO ── */
.hero-wrap {
    background: #1A1208;
    border-radius: 16px;
    padding: 3rem 2rem 2.4rem;
    text-align: center;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        90deg,
        transparent 0px, transparent 38px,
        rgba(232,136,26,0.06) 38px, rgba(232,136,26,0.06) 40px
    );
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: #E8881A;
    color: #1A1208;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 3px;
    margin-bottom: 1.1rem;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 4rem;
    font-weight: 900;
    line-height: 1.0;
    color: #FDF6EC;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.01em;
}
.hero-title em {
    font-style: italic;
    color: #E8881A;
}
.hero-sub {
    font-size: 0.97rem;
    color: #A89070;
    font-weight: 400;
    letter-spacing: 0.01em;
    margin-top: 0.5rem;
}
.hero-divider {
    width: 48px;
    height: 3px;
    background: #B5192B;
    border-radius: 2px;
    margin: 1.2rem auto 0;
}

/* ── Stats strip ── */
.stats-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 1.8rem;
}
.stat-card {
    background: #FFFFFF;
    border: 1px solid #E2D5C3;
    border-radius: 10px;
    padding: 1rem 0.6rem;
    text-align: center;
}
.stat-card:first-child  { border-top: 3px solid #B5192B; }
.stat-card:nth-child(2) { border-top: 3px solid #E8881A; }
.stat-card:nth-child(3) { border-top: 3px solid #1A6B35; }
.stat-card:last-child   { border-top: 3px solid #7C5CBF; }
.stat-num {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1.15;
    display: block;
    color: #1A1208;
}
.stat-desc {
    font-size: 0.66rem;
    color: #8A6F5E;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 3px;
    display: block;
}

/* ── Section label ── */
.sec-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #B5192B;
    display: block;
    margin-bottom: 0.5rem;
}

/* ── Input card wrapper ── */
.input-section {
    background: #FFFFFF;
    border: 1px solid #E2D5C3;
    border-radius: 14px;
    padding: 1.5rem 1.5rem 1.2rem;
    margin-bottom: 0.8rem;
}

/* ── Text area ── */
.stTextArea textarea {
    background-color: #FFF8F0 !important;
    color: #1A1208 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.98rem !important;
    font-weight: 400 !important;
    line-height: 1.8 !important;
    border: 1.5px solid #E2D5C3 !important;
    border-radius: 10px !important;
    padding: 1rem 1.1rem !important;
    caret-color: #B5192B;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.stTextArea textarea:focus {
    border-color: #B5192B !important;
    box-shadow: 0 0 0 3px rgba(181,25,43,0.10) !important;
}
.stTextArea textarea::placeholder {
    color: #C4AD98 !important;
    font-style: italic;
    font-size: 0.92rem !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #B5192B !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.68rem 1.4rem !important;
    transition: background 0.18s ease, transform 0.12s ease !important;
    box-shadow: 0 2px 12px rgba(181,25,43,0.28) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #961422 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(181,25,43,0.35) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

/* ── Secondary buttons ── */
.stButton > button:not([kind="primary"]) {
    background: #FFFFFF !important;
    color: #3D2B1F !important;
    font-size: 0.87rem !important;
    font-weight: 500 !important;
    border: 1px solid #E2D5C3 !important;
    border-radius: 8px !important;
    transition: border-color 0.18s, color 0.18s !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #B5192B !important;
    color: #B5192B !important;
}

/* ── Verdict — POSITIVE ── */
.verdict-pos {
    background: #F2FAF5;
    border: 1.5px solid #A8D5B8;
    border-left: 5px solid #1A6B35;
    border-radius: 12px;
    padding: 1.3rem 1.6rem;
    margin: 0.8rem 0 0.5rem;
}
.verdict-pos .v-label {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem;
    font-weight: 700;
    color: #1A6B35;
    display: block;
    line-height: 1.1;
    margin-bottom: 5px;
}
.verdict-pos .v-conf {
    font-size: 0.88rem;
    color: #2E7D4F;
    font-weight: 500;
}

/* ── Verdict — NEGATIVE ── */
.verdict-neg {
    background: #FEF2F3;
    border: 1.5px solid #F0AAAF;
    border-left: 5px solid #B5192B;
    border-radius: 12px;
    padding: 1.3rem 1.6rem;
    margin: 0.8rem 0 0.5rem;
}
.verdict-neg .v-label {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem;
    font-weight: 700;
    color: #B5192B;
    display: block;
    line-height: 1.1;
    margin-bottom: 5px;
}
.verdict-neg .v-conf {
    font-size: 0.88rem;
    color: #8F1422;
    font-weight: 500;
}

/* ── Confidence bar ── */
.stProgress > div > div > div > div {
    background: #B5192B !important;
    border-radius: 4px !important;
}
.stProgress > div > div > div {
    background: #F0E8DC !important;
    border-radius: 4px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E2D5C3 !important;
    border-radius: 10px !important;
    padding: 0.85rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    color: #8A6F5E !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
    color: #1A1208 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #FFFFFF !important;
    color: #3D2B1F !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    border: 1px solid #E2D5C3 !important;
}
.streamlit-expanderContent {
    background: #FFF8F0 !important;
    border: 1px solid #E2D5C3 !important;
    border-top: none !important;
}

/* ── Alerts & info ── */
.stAlert {
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
}
div[data-testid="stMarkdownContainer"] p {
    color: #3D2B1F !important;
    font-size: 0.95rem;
    line-height: 1.72;
}

/* ── Divider ── */
hr {
    border-color: #E2D5C3 !important;
    margin: 1.4rem 0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #FFF8F0 !important;
    border-right: 1px solid #E2D5C3 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #3D2B1F !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #B5192B !important; }

/* ── Caption ── */
.stCaption { color: #B4A090 !important; font-size: 0.75rem !important; }

/* ── Warning / info overrides ── */
div[data-baseweb="notification"] {
    background: #FFFFFF !important;
    border-radius: 10px !important;
}

/* ── Footer ── */
.footer-line {
    text-align: center;
    padding: 0.3rem 0 1.5rem;
    font-size: 0.72rem;
    color: #C4AD98;
    letter-spacing: 0.09em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RNN MODEL
# ─────────────────────────────────────────────
class SentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(SentimentRNN, self).__init__()
        self.rnn     = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc      = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.rnn(x)
        out = self.fc(h_n.squeeze(0))
        return self.sigmoid(out)


# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    errors = []
    try:
        with open("tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)
    except FileNotFoundError:
        errors.append("❌  `tfidf.pkl` not found — place it in the same folder as app.py")
        tfidf = None

    try:
        model = SentimentRNN(input_size=5000, hidden_size=128)
        model.load_state_dict(torch.load("rnn_model.pth", map_location=torch.device("cpu")))
        model.eval()
    except FileNotFoundError:
        errors.append("❌  `rnn_model.pth` not found — place it in the same folder as app.py")
        model = None
    except Exception as e:
        errors.append(f"❌  Error loading model: {str(e)}")
        model = None

    return model, tfidf, errors


@st.cache_resource
def load_nltk_resources():
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english")), PorterStemmer()


# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────
def clean_text(text: str, stop_words: set, stemmer: PorterStemmer) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>",        "", text)
    text = re.sub(r"http\S+",      "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens  = word_tokenize(text)
    cleaned = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(cleaned)


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_sentiment(review: str, model, tfidf, stop_words, stemmer):
    cleaned    = clean_text(review, stop_words, stemmer)
    vec        = tfidf.transform([cleaned]).toarray()
    tensor     = torch.tensor(vec, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        probability = model(tensor).item()
    is_positive = probability >= 0.5
    label       = "Positive" if is_positive else "Negative"
    confidence  = probability if is_positive else (1.0 - probability)
    return label, confidence, probability


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
def main():

    # ── Hero ──
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">🎬 &nbsp; Movie Sentiment Analysis</div>
        <div class="hero-title">Cine<em>Read</em></div>
        <p class="hero-sub">Paste any movie review — know the emotion in seconds.</p>
        <div class="hero-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats ──
    st.markdown("""
    <div class="stats-strip">
        <div class="stat-card">
            <span class="stat-num">50K</span>
            <span class="stat-desc">Reviews Trained</span>
        </div>
        <div class="stat-card">
            <span class="stat-num">87.6%</span>
            <span class="stat-desc">Accuracy</span>
        </div>
        <div class="stat-card">
            <span class="stat-num">5,000</span>
            <span class="stat-desc">TF-IDF Features</span>
        </div>
        <div class="stat-card">
            <span class="stat-num">RNN</span>
            <span class="stat-desc">Architecture</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load ──
    model, tfidf, load_errors = load_artifacts()
    stop_words, stemmer = load_nltk_resources()

    # ── Error state ──
    if load_errors:
        st.markdown("---")
        for err in load_errors:
            st.error(err)
        st.info(
            "**Quick Fix:**\n\n"
            "1. Run `RNN_Sentiment_Training.ipynb` in Google Colab\n"
            "2. Download `rnn_model.pth` and `tfidf.pkl`\n"
            "3. Place both files in the same folder as `app.py`\n"
            "4. Restart: `streamlit run app.py`"
        )
        st.stop()

    # ── Input ──
    st.markdown('<span class="sec-label">✍ Write or paste your review</span>', unsafe_allow_html=True)
    review_text = st.text_area(
        label="review_input",
        placeholder=(
            'e.g. "A breathtaking masterpiece — every frame glows with intention. '
            'The performances are career-best, and the score is unforgettable."'
        ),
        height=190,
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button(
            "🎬  Analyze Sentiment",
            use_container_width=True,
            type="primary"
        )

    # ── Result ──
    if predict_btn:
        if not review_text.strip():
            st.warning("⚠  Please enter a review before analyzing.")
        elif len(review_text.strip()) < 10:
            st.warning("⚠  Review too short — write at least one sentence.")
        else:
            with st.spinner("Reading the review…"):
                label, confidence, raw_prob = predict_sentiment(
                    review_text, model, tfidf, stop_words, stemmer
                )

            st.markdown("---")
            st.markdown('<span class="sec-label">📊 Verdict</span>', unsafe_allow_html=True)

            if label == "Positive":
                st.markdown(f"""
                <div class="verdict-pos">
                    <span class="v-label">😊 &nbsp; Positive Review</span>
                    <span class="v-conf">Confidence &nbsp;·&nbsp; {confidence * 100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-neg">
                    <span class="v-label">😞 &nbsp; Negative Review</span>
                    <span class="v-conf">Confidence &nbsp;·&nbsp; {confidence * 100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            st.progress(float(confidence))

            with st.expander("🔬  :Full Probability Breakdown: "):
                col_a, col_b = st.columns(2)
                col_a.metric("😊  Positive Score", f"{raw_prob * 100:.1f}%")
                col_b.metric("😞  Negative Score", f"{(1 - raw_prob) * 100:.1f}%")
                st.caption("Threshold: score ≥ 50% → Positive  |  score < 50% → Negative")

    # ── Sidebar ──
    st.sidebar.markdown("""
    <div style="text-align:center; padding:1.3rem 0 0.7rem;">
        <div style="font-family:'Playfair Display',serif !important;
                    font-size:1.4rem; font-weight:700; color:#B5192B; font-style:italic;">
            CineRead
        </div>
        <div style="font-size:0.68rem; color:#8A6F5E; letter-spacing:0.16em;
                    text-transform:uppercase; margin-top:3px; font-weight:600;">
            Sample Reviews
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<p style='font-size:0.83rem; color:#8A6F5E; margin-bottom:0.8rem;'>"
        "Load a sample to quickly test the model:</p>",
        unsafe_allow_html=True
    )

    positive_sample = (
        "An absolute gem of a movie! The storytelling was masterfully crafted, "
        "and the lead actor gave the performance of a lifetime. "
        "I left the theater feeling truly inspired and moved."
    )
    negative_sample = (
        "I honestly want my money back. The plot was full of holes, "
        "the pacing was painfully slow, and the dialogue felt incredibly forced. "
        "A massive disappointment from start to finish."
    )

    if st.sidebar.button("😊  Positive Sample", use_container_width=True):
        st.session_state["sample"] = positive_sample
        st.rerun()

    if st.sidebar.button("😞  Negative Sample", use_container_width=True):
        st.session_state["sample"] = negative_sample
        st.rerun()

    if "sample" in st.session_state:
        st.markdown("---")
        st.markdown('<span class="sec-label">💡 Sample Loaded</span>', unsafe_allow_html=True)
        st.info(st.session_state["sample"])
        del st.session_state["sample"]

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "<div class='footer-line'>"
        "🧠 RNN (PyTorch) &nbsp;·&nbsp; 📊 IMDB 50,000 Reviews &nbsp;·&nbsp; "
        "🔠 TF-IDF 5,000 Features &nbsp;·&nbsp; 🎯 87.62% Accuracy"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
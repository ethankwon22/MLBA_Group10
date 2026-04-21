import streamlit as st
import joblib
import boto3
import numpy as np
import textstat
import os
import tempfile

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Response Quality Scorer",
    page_icon="🤖",
    layout="wide"
)

# ── Load model from S3 (cached) ────────────────────────────────────────────
@st.cache_resource
def load_model():
    BUCKET = os.environ.get("S3_BUCKET", "llm-quality-group15")
    KEY    = "models/random_forest_model.pkl"
    try:
        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            s3.download_file(BUCKET, KEY, tmp.name)
            model = joblib.load(tmp.name)
        return model, f"✅ Model loaded from s3://{BUCKET}/{KEY}"
    except Exception as e:
        # Fallback: try loading locally for dev/testing
        if os.path.exists("random_forest_model.pkl"):
            model = joblib.load("random_forest_model.pkl")
            return model, "✅ Model loaded from local file"
        return None, f"❌ Model load failed: {e}"

# ── Feature extraction ─────────────────────────────────────────────────────
def extract_features(prompt: str, response: str) -> np.ndarray:
    r = response.strip()
    p = prompt.strip()
    words    = r.split()
    sentences = [s.strip() for s in r.replace('!','.').replace('?','.').split('.') if s.strip()]
    unique_words = set(w.lower() for w in words)

    char_len       = len(r)
    word_count     = len(words)
    sentence_count = max(len(sentences), 1)
    avg_word_len   = np.mean([len(w) for w in words]) if words else 0
    avg_sent_len   = word_count / sentence_count
    ttr            = len(unique_words) / word_count if word_count > 0 else 0

    has_bullets    = int('\n-' in r or '\n•' in r or '\n*' in r)
    has_numbered   = int(any(f'\n{i}.' in r for i in range(1, 10)))
    has_code       = int('```' in r or '`' in r)
    has_headers    = int('##' in r or '**' in r)
    newline_count  = r.count('\n')
    paragraph_count= max(len([x for x in r.split('\n\n') if x.strip()]), 1)

    try:
        flesch        = textstat.flesch_reading_ease(r)
        flesch_kincaid= textstat.flesch_kincaid_grade(r)
        gunning_fog   = textstat.gunning_fog(r)
    except Exception:
        flesch = flesch_kincaid = gunning_fog = 0.0

    prompt_word_count = len(p.split())
    rp_ratio = word_count / max(prompt_word_count, 1)

    return np.array([[
        char_len, word_count, sentence_count, avg_word_len, avg_sent_len, ttr,
        has_bullets, has_numbered, has_code, has_headers, newline_count, paragraph_count,
        flesch, flesch_kincaid, gunning_fog,
        prompt_word_count, rp_ratio
    ]])

def score_to_grade(score: float):
    if score >= 0.70:
        return "🟢 High Quality",   "#15803D", "This response is likely to be preferred by human evaluators."
    elif score >= 0.45:
        return "🟡 Moderate Quality","#CA8A04", "This response may win some comparisons but has room for improvement."
    else:
        return "🔴 Low Quality",    "#DC2626", "This response is unlikely to be preferred over a well-structured alternative."

# ── Main UI ────────────────────────────────────────────────────────────────
st.title("🤖 LLM Response Quality Scorer")
st.markdown(
    "Predict the **human preference win-rate score** of an LLM response "
    "using interpretable textual features — no secondary LLM required."
)

model, model_status = load_model()
st.caption(model_status)
st.divider()

# Input columns
col1, col2 = st.columns(2)
with col1:
    prompt_text = st.text_area(
        "📝 Prompt (user question)",
        height=160,
        placeholder="Enter the prompt / question that was given to the LLM...",
    )
with col2:
    response_text = st.text_area(
        "💬 LLM Response",
        height=160,
        placeholder="Paste the LLM response to evaluate here...",
    )

predict_btn = st.button("⚡ Score This Response", type="primary", use_container_width=True)

st.divider()

if predict_btn:
    if not prompt_text.strip() or not response_text.strip():
        st.warning("Please enter both a prompt and a response.")
    elif model is None:
        st.error("Model not loaded. Check S3 bucket configuration.")
    else:
        features = extract_features(prompt_text, response_text)
        score    = float(model.predict(features)[0])
        score    = max(0.0, min(1.0, score))  # clip to [0,1]
        grade, color, description = score_to_grade(score)

        # ── Result display ────────────────────────────────────────────────
        res_col1, res_col2, res_col3 = st.columns([1, 1, 2])

        with res_col1:
            st.metric("Win-Rate Score", f"{score:.3f}", help="0 = likely loses, 0.5 = tie, 1.0 = likely wins")
        with res_col2:
            st.markdown(f"### {grade}")
        with res_col3:
            st.info(description)

        st.progress(score)

        # ── Feature breakdown ─────────────────────────────────────────────
        with st.expander("🔍 Feature Breakdown", expanded=False):
            feat_names = [
                "char_len","word_count","sentence_count","avg_word_len","avg_sent_len","ttr",
                "has_bullets","has_numbered","has_code","has_headers","newline_count","paragraph_count",
                "flesch_reading_ease","flesch_kincaid_grade","gunning_fog",
                "prompt_word_count","response_to_prompt_ratio"
            ]
            feat_values = features[0].tolist()
            col_a, col_b = st.columns(2)
            half = len(feat_names) // 2
            with col_a:
                for n, v in zip(feat_names[:half], feat_values[:half]):
                    st.markdown(f"**{n}**: `{v:.3f}`")
            with col_b:
                for n, v in zip(feat_names[half:], feat_values[half:]):
                    st.markdown(f"**{n}**: `{v:.3f}`")

# ── Sidebar: model info ────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**Model:** Random Forest Regressor  
**Target:** Win-rate score (0 = loss, 0.5 = tie, 1.0 = win)  
**Features:** 17 textual features  
**Spearman ρ:** 0.2416 (best on test set)  
**Test RMSE:** 0.4083  
**Dataset:** LMSYS Chatbot Arena (33K conversations → 66K rows)

---
**CMU Tepper MSBA**  
Machine Learning for Business  
Group 15 — 2025
    """)
    st.markdown("---")
    st.markdown("**Feature categories:**")
    st.markdown("- 📏 Lexical (6 features)")
    st.markdown("- 🏗️ Structural (6 features)")
    st.markdown("- 📖 Readability (3 features)")
    st.markdown("- 🔗 Prompt-relative (2 features)")

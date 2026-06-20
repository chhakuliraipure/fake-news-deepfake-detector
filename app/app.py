"""
AI-Powered Misinformation Detection System
- Module 1: Fake News Text Classifier (TF-IDF + Logistic Regression)
- Module 2: Deepfake Video Detector (MesoNet CNN)

Run with: streamlit run app.py
"""
import os
import tempfile
import joblib
import streamlit as st

from video_predict import predict_video

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

st.set_page_config(page_title="AI Misinformation Detector", page_icon="🛡️", layout="centered")

# ---------- Load text model (cached) ----------
@st.cache_resource
def load_text_model():
    clf = joblib.load(os.path.join(MODEL_DIR, "fake_news_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    return clf, vectorizer


st.title("🛡️ AI-Powered Misinformation Detection System")
st.caption("College Project — Fake News (Text) Detection + Deepfake (Video) Detection")

tab1, tab2 = st.tabs(["📰 Fake News Detector", "🎥 Deepfake Video Detector"])

# ============== TAB 1: TEXT ==============
with tab1:
    st.subheader("Fake News Detection")
    st.write("Paste a news headline or full article text below to check if it's likely **REAL** or **FAKE**.")

    text_input = st.text_area("News text", height=200, placeholder="Paste article text here...")

    if st.button("Analyze Text", type="primary"):
        if not text_input.strip():
            st.warning("Please paste some text first.")
        else:
            clf, vectorizer = load_text_model()
            vec = vectorizer.transform([text_input])
            pred = clf.predict(vec)[0]
            proba = clf.predict_proba(vec)[0]
            classes = clf.classes_
            confidence = max(proba) * 100

            if pred == "FAKE":
                st.error(f"🚨 Prediction: **FAKE NEWS** ({confidence:.1f}% confidence)")
            else:
                st.success(f"✅ Prediction: **REAL NEWS** ({confidence:.1f}% confidence)")

            with st.expander("See probability breakdown"):
                for cls, p in zip(classes, proba):
                    st.write(f"{cls}: {p*100:.2f}%")

# ============== TAB 2: VIDEO ==============
with tab2:
    st.subheader("Deepfake Video Detection")
    st.write("Upload a short video clip with a visible face to check for deepfake manipulation artifacts.")

    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        st.video(video_file)

        if st.button("Analyze Video", type="primary"):
            with st.spinner("Extracting frames, detecting faces, running deepfake model..."):
                # save to temp file since OpenCV needs a path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name

                result = predict_video(tmp_path, max_frames=20)
                os.unlink(tmp_path)

            if "error" in result:
                st.warning(result["error"])
            else:
                if result["verdict"] == "FAKE":
                    st.error(f"🚨 Prediction: **DEEPFAKE DETECTED** ({result['confidence']}% confidence)")
                else:
                    st.success(f"✅ Prediction: **LIKELY AUTHENTIC** ({result['confidence']}% confidence)")

                st.caption(f"Analyzed {result['frames_analyzed']} frames | avg model score: {result['raw_avg_score']}")

                with st.expander("See per-frame scores"):
                    st.write(result["frame_scores"])
                    st.caption("Score closer to 1 = real face traits, closer to 0 = manipulation artifacts detected")

st.divider()
st.caption(
    "⚠️ Academic prototype. Text model trained on a ~6,300-article dataset (93% test accuracy). "
    "Video model uses pretrained MesoNet (Afchar et al., 2018) via transfer learning — not trained from scratch "
    "due to compute/dataset constraints. Real-world performance will vary."
)

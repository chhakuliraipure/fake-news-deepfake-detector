# AI-Powered Misinformation Detection System

A college project combining two detection modules in one dashboard:
1. **Fake News Detector** (text) — classifies news articles as REAL or FAKE
2. **Deepfake Video Detector** — flags videos with AI-manipulated faces

---

## 📁 Project Structure
```
project/
├── app/
│   ├── app.py                  # Main Streamlit dashboard (run this)
│   ├── train_text_model.py     # Trains the fake news classifier
│   ├── video_predict.py        # Deepfake video inference pipeline
│   └── mesonet_classifiers.py  # MesoNet model architecture
├── data/
│   └── fake_or_real_news.csv   # Training dataset (6,335 articles)
├── models/
│   ├── fake_news_model.joblib       # Trained text classifier
│   ├── tfidf_vectorizer.joblib      # Trained vectorizer
│   ├── text_model_metrics.json      # Accuracy/F1/confusion matrix (for report)
│   └── mesonet_weights/Meso4_DF.h5  # Pretrained deepfake detector weights
└── requirements.txt
```

## ⚙️ Setup (run these on your own laptop)

```bash
cd project
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
cd app
streamlit run app.py
```

This opens a browser window with two tabs — paste news text in one, upload a video in the other.

## 🔁 Retrain the Text Model (optional)
Already trained and saved, but if you want to rerun it:
```bash
cd app
python train_text_model.py
```

---

## 🧠 Methodology (for your report)

### Module 1: Fake News Detection (Text)
- **Dataset:** 6,335 labeled news articles (balanced: 3,171 REAL / 3,164 FAKE)
- **Preprocessing:** Combined title + article body, removed stopwords
- **Feature extraction:** TF-IDF vectorization (unigrams + bigrams, top 50,000 features)
- **Model:** Logistic Regression
- **Result:** **92.98% test accuracy**, F1-score 0.93 (see `models/text_model_metrics.json` for full breakdown and confusion matrix)
- **Why this approach:** TF-IDF + Logistic Regression is fast to train, interpretable (you can show which words drive predictions), and a well-established baseline in fake news detection literature — appropriate given time constraints, while still being defensible academically.

### Module 2: Deepfake Video Detection
- **Approach:** Transfer learning — used a pretrained **MesoNet (Meso4)** architecture (Afchar et al., 2018, *"MesoNet: a Compact Facial Video Forgery Detection Network"*), rather than training a CNN from scratch on a deepfake dataset (e.g. FaceForensics++), which would require significant GPU time and large dataset downloads not feasible in this timeframe.
- **Pipeline:**
  1. Extract ~20 evenly-spaced frames from the uploaded video
  2. Detect the face in each frame using OpenCV's Haar Cascade classifier
  3. Crop and resize each face to 256×256
  4. Run each face crop through the MesoNet CNN, which outputs a score (closer to 1 = authentic, closer to 0 = manipulated)
  5. Average all frame scores → final video-level verdict
- **Validated on sample real/fake face images:** correctly classified with >95% confidence on both classes (see test run in project history)

### Limitations (good to mention in your report — shows critical thinking)
- Text model is trained on one dataset; accuracy on real-world/current news (different writing style, new topics) will likely be lower. This is a known generalization challenge in fake news detection research.
- Deepfake detector uses a pretrained model not fine-tuned on your specific test videos; works best on clear, front-facing single-face videos. Performance on heavily compressed or low-resolution video (e.g. re-uploaded social media clips) may degrade, since compression artifacts can resemble manipulation artifacts.
- Both modules are prototypes for demonstrating the pipeline/architecture, not production-grade systems.

---

## 🎤 Possible Viva/Demo Questions & Answers

**Q: Did you train the deepfake model yourself?**
A: We used transfer learning with a pretrained MesoNet model rather than training from scratch, since deepfake datasets (FaceForensics++, DFDC) are very large (100GB+) and training a CNN from scratch requires significant GPU resources. This is a standard and accepted practice in applied ML projects — we built and validated the full inference pipeline (frame extraction, face detection, aggregation) ourselves.

**Q: How does the text model decide real vs fake?**
A: It converts text into TF-IDF features (word/phrase importance scores) and uses Logistic Regression to learn which words/phrases are statistically associated with fake vs real articles in the training data.

**Q: What would you improve with more time?**
A: Fine-tune a transformer-based model (e.g. BERT) for text for better generalization; fine-tune the video model on a larger labeled deepfake dataset; add explainability (highlight which words/frames triggered the prediction).

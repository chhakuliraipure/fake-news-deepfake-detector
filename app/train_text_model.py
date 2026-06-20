"""
Fake News Detection - Text Classifier Training Script
Approach: TF-IDF vectorization + Logistic Regression / PassiveAggressive
Dataset: fake_or_real_news.csv (6335 labeled news articles)
"""
import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fake_or_real_news.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])

    # Combine title + text for richer signal
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # TF-IDF Vectorizer
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        min_df=2,
        ngram_range=(1, 2),
        max_features=50000,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression (primary model)
    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_).tolist()

    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)

    # Save model + vectorizer
    joblib.dump(clf, os.path.join(MODEL_DIR, "fake_news_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

    # Save metrics for the report
    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": clf.classes_.tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(os.path.join(MODEL_DIR, "text_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel + vectorizer saved to {MODEL_DIR}")
    print("Metrics saved to text_model_metrics.json (use this in your report)")

if __name__ == "__main__":
    main()

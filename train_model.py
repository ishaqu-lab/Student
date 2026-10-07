from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from preprocess import preprocess_text, ensure_nltk_resources

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "default_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

REQUIRED_COLUMNS = ["ID", "Feedback Text", "Category", "Sentiment", "Date"]
VALID_CATEGORIES = [
    "Teacher Feedback",
    "Course Content",
    "Examination Pattern",
    "Laboratory",
    "Library Facilities",
]
VALID_SENTIMENTS = ["Positive", "Neutral", "Negative"]

def main():
    ensure_nltk_resources()
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["Feedback Text", "Sentiment"]).copy()
    df["Feedback Text"] = df["Feedback Text"].astype(str).str.strip()

    duplicate_count = int(df["Feedback Text"].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"Dataset contains {duplicate_count} duplicate feedback entries. "
            "Remove duplicates before training."
        )
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Sentiment"] = df["Sentiment"].astype(str).str.strip()

    if not df["Category"].isin(VALID_CATEGORIES).all():
        raise ValueError("Dataset contains unsupported categories.")
    if not df["Sentiment"].isin(VALID_SENTIMENTS).all():
        raise ValueError("Dataset contains unsupported sentiment labels.")

    df["Processed Text"] = df["Feedback Text"].apply(preprocess_text)

    X = df["Processed Text"]
    y = df["Sentiment"]

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df.index,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=3,
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB(alpha=1.0)),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
    }

    report = classification_report(
        y_test, predictions, labels=VALID_SENTIMENTS,
        output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_test, predictions, labels=VALID_SENTIMENTS).tolist()

    # 5-fold cross-validation on the full labelled dataset.
    cv_scores = cross_val_score(
        pipeline, X, y, cv=5, scoring="accuracy"
    )

    metrics["cv_mean_accuracy"] = float(cv_scores.mean())
    metrics["cv_std_accuracy"] = float(cv_scores.std())

    joblib.dump(pipeline, MODEL_DIR / "sentiment_model.pkl")

    metadata = {
        "categories": VALID_CATEGORIES,
        "sentiments": VALID_SENTIMENTS,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": VALID_SENTIMENTS,
        "dataset_records": int(len(df)),
        "unique_feedback_records": int(df["Feedback Text"].nunique()),
        "duplicate_feedback_records": int(duplicate_count),
        "category_distribution": df["Category"].value_counts().to_dict(),
        "sentiment_distribution": df["Sentiment"].value_counts().to_dict(),
        "model": "Multinomial Naive Bayes",
        "vectorizer": "TF-IDF",
        "preprocessing": [
            "lowercasing",
            "tokenization",
            "punctuation removal",
            "stopword removal",
            "WordNet lemmatization",
            "negation preservation"
        ],
        "random_state": 42,
        "test_size": 0.20,
    }

    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save test predictions for reproducibility.
    test_results = df.loc[test_idx, ["ID", "Feedback Text", "Category", "Sentiment", "Date"]].copy()
    test_results["Predicted Sentiment"] = predictions
    test_results.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    print("Model training completed.")
    print(f"Training records: {len(y_train)}")
    print(f"Testing records: {len(y_test)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"5-fold CV accuracy: {metrics['cv_mean_accuracy']:.4f}")

if __name__ == "__main__":
    main()

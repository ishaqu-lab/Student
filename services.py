from pathlib import Path
import json
import io
import pandas as pd
import joblib
from flask import current_app
from werkzeug.utils import secure_filename

from preprocess import preprocess_text

VALID_CATEGORIES = [
    "Teacher Feedback",
    "Course Content",
    "Examination Pattern",
    "Laboratory",
    "Library Facilities",
]
VALID_SENTIMENTS = ["Positive", "Neutral", "Negative"]

# Sentiment is deliberately NOT required for uploaded data.
# It is produced by the trained classifier.
REQUIRED_UPLOAD_COLUMNS = ["ID", "Feedback Text", "Category", "Date"]

def model_path():
    return Path(current_app.root_path) / "models" / "sentiment_model.pkl"

def metadata_path():
    return Path(current_app.root_path) / "models" / "metadata.json"

def load_model():
    path = model_path()
    if not path.exists():
        return None
    return joblib.load(path)

def load_metadata():
    path = metadata_path()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_csv(file):
    if not file or not file.filename:
        return False, "Please select a CSV file."

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".csv"):
        return False, "Only CSV files are allowed."

    file.stream.seek(0)
    content = file.stream.read()
    file.stream.seek(0)

    max_bytes = current_app.config.get("MAX_CONTENT_LENGTH", 10 * 1024 * 1024)
    if len(content) > max_bytes:
        return False, "The uploaded file exceeds the 10 MB limit."

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        return False, f"Unable to read the CSV file: {exc}"

    # Remove accidental spaces around column names.
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in df.columns]
    if missing:
        return False, "Missing required columns: " + ", ".join(missing)

    if df.empty:
        return False, "The uploaded CSV is empty."

    # Validate the fields that are actually supplied by a new feedback file.
    if df["Feedback Text"].isna().any():
        return False, "Feedback Text contains empty values."

    df["Feedback Text"] = df["Feedback Text"].astype(str).str.strip()
    if (df["Feedback Text"] == "").any():
        return False, "Feedback Text contains empty values."

    if not df["Category"].isin(VALID_CATEGORIES).all():
        return False, "The CSV contains an unsupported category."

    # A Sentiment column may be present in a labelled file, but it is not
    # trusted for analysis. The application predicts the sentiment itself.
    if "Sentiment" in df.columns:
        df = df.drop(columns=["Sentiment"])

    return True, df

def predict_feedback(texts):
    """Predict sentiment for raw feedback text using the saved TF-IDF + NB model."""
    model = load_model()
    if model is None:
        raise RuntimeError("Model has not been trained. Run train_model.py first.")

    raw_texts = pd.Series(texts).astype(str)
    processed_texts = raw_texts.apply(preprocess_text)

    if (processed_texts.str.strip() == "").any():
        raise ValueError("One or more feedback entries contain no usable words after preprocessing.")

    predictions = model.predict(processed_texts)
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_texts).max(axis=1)
    return predictions, probabilities

def apply_filters(df, category=None, sentiment=None, search=None):
    out = df.copy()
    if category and category != "All Categories":
        out = out[out["Category"] == category]
    if sentiment and sentiment != "All Sentiments":
        out = out[out["Sentiment"] == sentiment]
    if search:
        needle = str(search).strip().lower()
        if needle:
            mask = (
                out["Feedback Text"].astype(str).str.lower().str.contains(needle, regex=False)
                | out["Category"].astype(str).str.lower().str.contains(needle, regex=False)
                | out["Sentiment"].astype(str).str.lower().str.contains(needle, regex=False)
            )
            out = out[mask]
    return out

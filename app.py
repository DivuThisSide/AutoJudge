from flask import Flask, render_template, request
import joblib
import numpy as np
import re
import math
from scipy.sparse import hstack

app = Flask(__name__)

# ======================
# LOAD MODELS
# ======================
clf = joblib.load("classifier.pkl")
reg = joblib.load("regressor.pkl")
tfidf = joblib.load("tfidf.pkl")
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")

# ======================
# TEXT CLEANING (SAME AS TRAINING)
# ======================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s\+\-\*/=<>\(\)\[\]\{\}\^\%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ======================
# FEATURE EXTRACTION (MATCH TRAINING)
# ======================
def extract_features(text):
    feats = {}

    words = text.split()
    feats["text_length"] = len(text)
    feats["word_count"] = len(words)
    feats["unique_word_ratio"] = len(set(words)) / max(len(words), 1)
    feats["sentence_count"] = max(len(re.split(r'[.!?]+', text)), 1)
    feats["avg_sentence_length"] = feats["word_count"] / feats["sentence_count"]

    feats["math_symbols"] = sum(text.count(c) for c in "+-*/=<>()[]{}^%")
    feats["brackets_count"] = sum(text.count(c) for c in "()[]{}")

    nums = [int(n) for n in re.findall(r"\b\d+\b", text)]
    feats["num_count"] = len(nums)
    feats["max_number"] = max(nums) if nums else 0
    feats["avg_number"] = np.mean(nums) if nums else 0
    feats["std_number"] = np.std(nums) if len(nums) > 1 else 0
    feats["log_max_number"] = math.log1p(feats["max_number"])

    feats["has_large_n"] = int(feats["max_number"] >= 100000)
    feats["has_huge_n"] = int(feats["max_number"] >= 1000000)

    patterns = {
        "adv_dp": r"(dp|dynamic programming)",
        "adv_graph": r"(dijkstra|shortest path|graph)",
        "med_greedy": r"(greedy)",
        "med_binary": r"(binary search)",
        "basic_math": r"(sum|product|count)"
    }

    for key, pat in patterns.items():
        feats[key] = int(bool(re.search(pat, text)))

    feats["total_advanced"] = feats["adv_dp"] + feats["adv_graph"]
    feats["total_medium"] = feats["med_greedy"] + feats["med_binary"]
    feats["total_basic"] = feats["basic_math"]

    total = feats["total_advanced"] + feats["total_medium"] + feats["total_basic"]
    feats["ratio_advanced"] = feats["total_advanced"] / (total + 1)
    feats["ratio_medium"] = feats["total_medium"] / (total + 1)
    feats["ratio_basic"] = feats["total_basic"] / (total + 1)

    return feats

# ======================
# ROUTES
# ======================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None
    tags = []

    if request.method == "POST":
        text = (
            request.form["title"] + " " +
            request.form["description"] + " " +
            request.form["input_desc"] + " " +
            request.form["output_desc"]
        )

        text = clean_text(text)

        # TF-IDF
        X_text = tfidf.transform([text])

        # Manual features
        feats = extract_features(text)
        X_extra = np.array([[feats.get(col, 0) for col in feature_cols]])
        X_extra = scaler.transform(X_extra)

        X = hstack([X_text, X_extra])

        # Prediction
        class_pred = clf.predict(X)[0]
        result = le.inverse_transform([class_pred])[0].capitalize()

        score = round(float(reg.predict(X)[0]), 2)

        # Tags (UI explanation)
        for k, v in feats.items():
            if k.startswith(("adv_", "med_", "basic_")) and v == 1:
                tags.append(k.replace("_", " ").upper())

    return render_template(
        "index.html",
        result=result,
        score=score,
        tags=tags
    )

if __name__ == "__main__":
    app.run(debug=True)

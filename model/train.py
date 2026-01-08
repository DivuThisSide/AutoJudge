import json
import re
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.sparse import hstack

# ======================
# 1. LOAD DATASET
# ======================
data = []
with open(r"./data/problems_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

df = pd.DataFrame(data)
print("Dataset shape:", df.shape)

# ======================
# 2. TEXT CLEANING
# ======================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s\+\-\*/=<>\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ======================
# 3. COMBINE TEXT FIELDS
# ======================
df["text"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["input_description"].fillna("") + " " +
    df["output_description"].fillna("")
)

df["text"] = df["text"].apply(clean_text)

# ======================
# 4. TARGET ENCODING
# ======================
le = LabelEncoder()
df["class_label"] = le.fit_transform(df["problem_class"])

y_class = df["class_label"]
y_reg = df["problem_score"]


# ======================
# 5. FEATURE ENGINEERING
# ======================

# ---- TF-IDF (already strong) ----
tfidf = TfidfVectorizer(
    max_features=8000,
    analyzer="char_wb",
    ngram_range=(3, 5)
)

X_text = tfidf.fit_transform(df["text"])

# ---- Basic numeric features ----
df["text_length"] = df["text"].apply(len)

df["math_symbols"] = df["text"].apply(
    lambda x: sum(x.count(c) for c in "+-*/=<>()")
)

# ---- Algorithm keyword FREQUENCY ----
algo_keywords = [
    "dp", "dynamic programming",
    "graph", "tree",
    "greedy",
    "binary search",
    "dfs", "bfs",
    "bitmask",
    "segment tree",
    "fenwick",
    "shortest path",
    "dijkstra"
    # "topological",
    # "flow",
    # "matching"
]

for kw in algo_keywords:
    df[f"freq_{kw.replace(' ', '_')}"] = df["text"].str.count(kw)

# ---- Time / efficiency indicators ----
complexity_terms = [
    "o(n)", "o(n log n)", "o(n^2)", "o(n^3)",
    "log n", "n log n",
    "time complexity",
    "optimize", "efficient"
]

for term in complexity_terms:
    df[f"has_{term.replace(' ', '_').replace('^','')}"] = \
        df["text"].str.contains(term).astype(int)

def max_constraint(text):
    nums = re.findall(r"\d+", text)
    nums = [int(n) for n in nums if len(n) <= 7]
    return max(nums) if nums else 0

# ---- Constraint magnitude (VERY IMPORTANT) ----
df["max_constraint"] = df["text"].apply(max_constraint)
df["has_large_constraints"] = (df["max_constraint"] >= 100000).astype(int)

# ---- Combine extra features ----
extra_features = (
    ["text_length", "math_symbols", "max_constraint", "has_large_constraints"] +
    [f"freq_{kw.replace(' ', '_')}" for kw in algo_keywords] +
    [f"has_{term.replace(' ', '_').replace('^','')}" for term in complexity_terms]
)

X_extra = df[extra_features].values

# ---- Final feature matrix ----
X = hstack([X_text, X_extra])

# ======================
# 6. CLASSIFICATION MODEL
# ======================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

# clf = LogisticRegression(
#     max_iter=1000,
#     multi_class="auto"
# )
clf = LogisticRegression(
    max_iter=3000,
    n_jobs=-1
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("\n--- Classification ---")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# ======================
# 7. REGRESSION MODEL
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42, stratify=y_class
)

reg = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)
reg.fit(X_train, y_train)

pred = reg.predict(X_test)
print("\n--- Regression ---")
print("MAE:", mean_absolute_error(y_test, pred))
rmse = mean_squared_error(y_test, pred) ** 0.5
print("RMSE:", rmse)

# ======================
# 8. SAVE MODELS
# ======================
joblib.dump(clf, "model/classifier.pkl")
joblib.dump(reg, "model/regressor.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("\nModels saved successfully.")

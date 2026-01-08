import json
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import hstack

# ======================
# 1. LOAD DATASET
# ======================
data = []
with open("./data/problems_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

df = pd.DataFrame(data)

# ======================
# 2. TEXT CLEANING
# ======================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s\+\-\*/=<>()]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["text"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["input_description"].fillna("") + " " +
    df["output_description"].fillna("")
).apply(clean_text)

# ======================
# 3. LABEL ENCODING
# ======================
le = LabelEncoder()
df["class_label"] = le.fit_transform(df["problem_class"])
y_class = df["class_label"]

# ======================
# 4. FEATURE ENGINEERING
# ======================
tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=7000
)
X_text = tfidf.fit_transform(df["text"])

def max_constraint(text):
    nums = re.findall(r"\d+", text)
    nums = [int(n) for n in nums if len(n) <= 7]
    return max(nums) if nums else 0

df["text_len"] = df["text"].apply(len)
df["math_ops"] = df["text"].apply(lambda x: sum(x.count(c) for c in "+-*/=<>()"))
df["max_constraint"] = df["text"].apply(max_constraint)
df["large_constraint"] = (df["max_constraint"] >= 100000).astype(int)

algo_keywords = [
    "dp", "dynamic programming", "graph", "tree",
    "dfs", "bfs", "greedy", "binary search",
    "segment tree", "dijkstra"
]

for kw in algo_keywords:
    df[f"freq_{kw.replace(' ', '_')}"] = df["text"].str.count(kw)

extra_cols = (
    ["text_len", "math_ops", "max_constraint", "large_constraint"] +
    [f"freq_{kw.replace(' ', '_')}" for kw in algo_keywords]
)

X_extra = df[extra_cols].values
X = hstack([X_text, X_extra])

# ======================
# 5. CLASSIFICATION
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

clf = LogisticRegression(
    max_iter=3000,
    n_jobs=-1,
    class_weight="balanced",
    C=0.7
)

clf.fit(X_train, y_train)

# (Optional evaluation – keep for report)
pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, pred)
clf_cm = confusion_matrix(y_test, pred)
print(clf_accuracy)
print("\n")
print(clf_cm)
print("\n")

# ======================
# 6. HYBRID RESIDUAL REGRESSION
# ======================
class_base_score = (
    df.groupby("class_label")["problem_score"]
    .mean()
    .to_dict()
)

y_residual = df.apply(
    lambda r: r["problem_score"] - class_base_score[r["class_label"]],
    axis=1
)

Xr_train, Xr_test, yr_train, yr_test, cls_train, cls_test = train_test_split(
    X,
    y_residual,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

residual_reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

residual_reg.fit(Xr_train, yr_train)

pred_residual = residual_reg.predict(Xr_test)

final_pred = [
    class_base_score[cls] + pred_residual[i]
    for i, cls in enumerate(cls_test)
]

final_pred = np.clip(final_pred, 1.0, 5.0)

true_scores = df.loc[yr_test.index, "problem_score"]

reg_mae = mean_absolute_error(true_scores, final_pred)
reg_rmse = mean_squared_error(true_scores, final_pred) ** 0.5
print(reg_mae)
print("\n")
print(reg_rmse)
print("\n")

# ======================
# 7. CONFIDENCE + PERCENTILES
# ======================
residual_std = float(np.std(y_residual))

score_distributions = {
    cls: df[df["class_label"] == cls]["problem_score"].values
    for cls in np.unique(y_class)
}

# ======================
# 8. SAVE MODELS
# ======================
joblib.dump(clf, "model/classifier.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")
joblib.dump(le, "model/label_encoder.pkl")
joblib.dump(extra_cols, "model/extra_cols.pkl")

joblib.dump(residual_reg, "model/residual_regressor.pkl")
joblib.dump(class_base_score, "model/class_base_score.pkl")
joblib.dump(residual_std, "model/residual_std.pkl")
joblib.dump(score_distributions, "model/score_distributions.pkl")

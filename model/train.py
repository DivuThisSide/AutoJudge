import json
import re
import math
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("AutoJudge - Training ML Models")
print("="*60)

# Load dataset
data = []
with open("./data/problems_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))
df = pd.DataFrame(data)
print(f"Dataset loaded: {df.shape[0]} problems")

# Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s\+\-\*/=<>\(\)\[\]\{\}\^\%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Combine text fields (title weighted 2x)
df["text"] = (
    df["title"].fillna("") + " " +
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["input_description"].fillna("") + " " +
    df["output_description"].fillna("")
)
df["text"] = df["text"].apply(clean_text)

# Feature engineering
def extract_features(df):
    # Basic text statistics
    df["text_length"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["unique_word_ratio"] = df["text"].apply(lambda x: len(set(x.split())) / max(len(x.split()), 1))
    df["sentence_count"] = df["text"].apply(lambda x: len(re.split(r'[.!?]+', x)))
    df["avg_sentence_length"] = df["word_count"] / df["sentence_count"].replace(0, 1)
    df["math_symbols"] = df["text"].apply(lambda x: sum(x.count(c) for c in "+-*/=<>()[]{}^%"))
    df["brackets_count"] = df["text"].apply(lambda x: sum(x.count(c) for c in "()[]{}"))

    # Number analysis
    def analyze_numbers(text):
        nums = [int(n) for n in re.findall(r'\b\d+\b', text)]
        if not nums:
            return 0, 0, 0, 0
        return len(nums), max(nums), np.mean(nums), np.std(nums) if len(nums) > 1 else 0
    
    df["num_count"], df["max_number"], df["avg_number"], df["std_number"] = zip(*df["text"].apply(analyze_numbers))
    df["log_max_number"] = df["max_number"].apply(lambda x: math.log1p(x))
    df["has_large_n"] = (df["max_number"] >= 100000).astype(int)
    df["has_huge_n"] = (df["max_number"] >= 1000000).astype(int)

    # Algorithm pattern detection
    advanced_patterns = {
        'dp_explicit': r'\b(dynamic\s*programming|dp\s*solution|dp\s*table|memoization|tabulation)\b',
        'graph_shortest': r'\b(dijkstra|bellman|floyd|shortest\s*path)\b',
        'graph_spanning': r'\b(kruskal|prim|spanning\s*tree|mst)\b',
        'graph_flow': r'\b(max\s*flow|min\s*cut|ford\s*fulkerson)\b',
        'advanced_ds': r'\b(segment\s*tree|fenwick|trie|suffix)\b',
        'string_algo': r'\b(kmp|rabin\s*karp|z\s*algorithm|manacher)\b',
        'number_theory': r'\b(modular\s*arithmetic|gcd|lcm|coprime|prime\s*factorization)\b',
    }
    for key, pattern in advanced_patterns.items():
        df[f"adv_{key}"] = df["text"].str.contains(pattern, regex=True).astype(int)

    medium_patterns = {
        'two_pointer': r'\b(two\s*pointer|sliding\s*window)\b',
        'binary_search': r'\b(binary\s*search|bisect)\b',
        'greedy': r'\b(greedy|locally\s*optimal)\b',
        'backtrack': r'\b(backtrack|permutation|combination)\b',
        'bfs_dfs': r'\b(bfs|dfs|breadth\s*first|depth\s*first)\b',
        'graph_basic': r'\b(graph|tree|node|edge|adjacent)\b',
        'sorting_complex': r'\b(merge\s*sort|quick\s*sort|heap\s*sort)\b',
    }
    for key, pattern in medium_patterns.items():
        df[f"med_{key}"] = df["text"].str.contains(pattern, regex=True).astype(int)

    basic_patterns = {
        'simple_array': r'\b(array\s*manipulation|array\s*rotation)\b',
        'simple_string': r'\b(reverse\s*string|palindrome\s*check)\b',
        'simple_math': r'\b(sum|product|average|count)\b',
        'sorting_basic': r'\b(sort\s*array|sort\s*list)\b',
        'iteration': r'\b(iterate|loop|for\s*each)\b',
    }
    for key, pattern in basic_patterns.items():
        df[f"basic_{key}"] = df["text"].str.contains(pattern, regex=True).astype(int)

    # Problem characteristics
    df["multi_test_cases"] = df["text"].str.contains(r'test\s*case|multiple\s*case|t\s*test').astype(int)
    df["single_line_input"] = df["text"].str.contains(r'single\s*line').astype(int)
    df["matrix_input"] = df["text"].str.contains(r'matrix|grid|2d\s*array').astype(int)
    df["has_constraints"] = df["text"].str.contains(r'constraint|limit|bound').astype(int)
    df["time_limit"] = df["text"].str.contains(r'time\s*limit|time\s*complexity').astype(int)
    df["space_limit"] = df["text"].str.contains(r'space\s*limit|space\s*complexity|memory').astype(int)
    df["asks_optimal"] = df["text"].str.contains(r'optimal|minimum|maximum|best|efficient').astype(int)
    df["asks_count"] = df["text"].str.contains(r'how\s*many|count\s*the|number\s*of').astype(int)
    df["asks_yes_no"] = df["text"].str.contains(r'possible|impossible|can\s*you|is\s*it\s*possible').astype(int)
    df["avg_word_length"] = df["text"].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
    df["long_words_count"] = df["text"].apply(lambda x: sum(1 for w in x.split() if len(w) > 10))

    # Aggregate features
    df["total_advanced"] = sum(df[col] for col in df.columns if col.startswith("adv_"))
    df["total_medium"] = sum(df[col] for col in df.columns if col.startswith("med_"))
    df["total_basic"] = sum(df[col] for col in df.columns if col.startswith("basic_"))
    
    total_patterns = df["total_advanced"] + df["total_medium"] + df["total_basic"]
    df["ratio_advanced"] = df["total_advanced"] / (total_patterns + 1)
    df["ratio_medium"] = df["total_medium"] / (total_patterns + 1)
    df["ratio_basic"] = df["total_basic"] / (total_patterns + 1)
    
    return df

df = extract_features(df)
print(f"Features extracted: {len([c for c in df.columns if any(c.startswith(p) for p in ['adv_','med_','basic_'])])} patterns")

# Prepare targets
le = LabelEncoder()
y_class = le.fit_transform(df["problem_class"])
y_reg = df["problem_score"].values

# TF-IDF vectorization
tfidf = TfidfVectorizer(
    max_features=3000,
    stop_words="english",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)
X_text = tfidf.fit_transform(df["text"])

# Combine all features
feature_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in 
    ['adv_', 'med_', 'basic_', 'total_', 'ratio_'])] + [
    "text_length", "word_count", "unique_word_ratio", "sentence_count", 
    "avg_sentence_length", "math_symbols", "brackets_count",
    "num_count", "max_number", "avg_number", "std_number", "log_max_number",
    "has_large_n", "has_huge_n", "multi_test_cases", "single_line_input", 
    "matrix_input", "has_constraints", "time_limit", "space_limit",
    "asks_optimal", "asks_count", "asks_yes_no", "avg_word_length", "long_words_count"
]

scaler = StandardScaler()
X_manual = scaler.fit_transform(df[feature_cols])
X = hstack([X_text, X_manual])

print(f"Total features: {X.shape[1]} (TF-IDF: {X_text.shape[1]}, Manual: {len(feature_cols)})")

# Train-test split with SMOTE balancing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

smote = SMOTE(random_state=42, k_neighbors=3)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"Training samples: {X_train_bal.shape[0]} (balanced with SMOTE)")

# Train models
print("\nTraining models...")

# Gradient Boosting
clf_gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.08, max_depth=7,
    min_samples_split=15, min_samples_leaf=5, subsample=0.85,
    max_features='sqrt', random_state=42
)
clf_gb.fit(X_train_bal, y_train_bal)
gb_acc = accuracy_score(y_test, clf_gb.predict(X_test))

# Random Forest
clf_rf = RandomForestClassifier(
    n_estimators=300, max_depth=25, min_samples_split=15,
    min_samples_leaf=4, max_features='sqrt', class_weight='balanced',
    random_state=42, n_jobs=-1
)
clf_rf.fit(X_train_bal, y_train_bal)
rf_acc = accuracy_score(y_test, clf_rf.predict(X_test))

# SVM
clf_svm = SVC(
    kernel='rbf', C=10, gamma='scale', class_weight='balanced',
    probability=True, random_state=42
)
clf_svm.fit(X_train_bal, y_train_bal)
svm_acc = accuracy_score(y_test, clf_svm.predict(X_test))

# Ensemble prediction
weights = np.array([gb_acc**2, rf_acc**2, svm_acc**2])
weights = weights / weights.sum()

ensemble_proba = (
    weights[0] * clf_gb.predict_proba(X_test) +
    weights[1] * clf_rf.predict_proba(X_test) +
    weights[2] * clf_svm.predict_proba(X_test)
)
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

# Select best model
accuracies = [('GB', gb_acc), ('RF', rf_acc), ('SVM', svm_acc), ('Ensemble', ensemble_acc)]
best_name, best_acc = max(accuracies, key=lambda x: x[1])
best_pred = ensemble_pred if best_name == 'Ensemble' else {
    'GB': clf_gb, 'RF': clf_rf, 'SVM': clf_svm
}[best_name].predict(X_test)

print(f"Model accuracies: GB={gb_acc:.4f}, RF={rf_acc:.4f}, SVM={svm_acc:.4f}, Ensemble={ensemble_acc:.4f}")
print(f"Best model: {best_name}")

# Classification metrics
print("\n" + "="*60)
print("CLASSIFICATION RESULTS")
print("="*60)
print(f"\nAccuracy: {best_acc:.4f}\n")
print(classification_report(y_test, best_pred, target_names=le.classes_, digits=4))

cm = confusion_matrix(y_test, best_pred)
print("Confusion Matrix:")
print(cm)
print("\nPer-class Recall:")
for i, cls in enumerate(le.classes_):
    recall = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"  {cls.capitalize():8s}: {recall:.2%}")

# Regression model
regressors = {}
X_tr, X_te, y_tr, y_te, y_class_tr, y_class_te = train_test_split(
    X, y_reg, y_class,
    test_size=0.2,
    random_state=42
)

for cls in np.unique(y_class_tr):
    idx = np.where(y_class_tr == cls)[0]

    reg_c = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        random_state=42,
        n_jobs=-1
    )
    reg_c.fit(X_tr[idx], y_tr[idx])
    regressors[int(cls)] = reg_c

y_pred = []

for i in range(len(y_te)):
    cls = y_class_te[i]
    reg = regressors[int(cls)]
    pred = reg.predict(X_te[i])[0]
    pred = np.clip(pred, 1.0, 5.0)
    y_pred.append(pred)

y_pred = np.array(y_pred)

mae = mean_absolute_error(y_te, y_pred)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))

print("\n" + "="*60)
print("REGRESSION RESULTS (CLASS-CONDITIONED)")
print("="*60)
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

joblib.dump(regressors, "regressors_by_class.pkl")

# save models
clf = clf_gb  # Save best individual model for inference
joblib.dump(clf, "classifier.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

if best_name == 'Ensemble':
    joblib.dump([clf_gb, clf_rf, clf_svm], "ensemble_models.pkl")
    joblib.dump(weights, "ensemble_weights.pkl")

print(f"Training complete! Final accuracy: {best_acc:.2%}")
print("="*60)
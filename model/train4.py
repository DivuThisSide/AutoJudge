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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.sparse import hstack, vstack
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("AutoJudge - Programming Problem Difficulty Predictor")
print("="*60)

# ======================
# 1. LOAD DATASET
# ======================
print("\n[1/7] Loading dataset...")
data = []
with open("./data/problems_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

df = pd.DataFrame(data)
print(f"  ✓ Loaded {df.shape[0]} problems")
print(f"  ✓ Distribution: Easy={len(df[df['problem_class']=='easy'])}, "
      f"Medium={len(df[df['problem_class']=='medium'])}, "
      f"Hard={len(df[df['problem_class']=='hard'])}")

# ======================
# 2. TEXT PREPROCESSING
# ======================
print("\n[2/7] Preprocessing text...")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Weight title more (appears 3x)
df["text"] = (
    df["title"].fillna("") + " " +
    df["title"].fillna("") + " " +
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["input_description"].fillna("") + " " +
    df["output_description"].fillna("")
)
df["text"] = df["text"].apply(clean_text)
print("  ✓ Text preprocessed")

# ======================
# 3. FEATURE ENGINEERING
# ======================
print("\n[3/7] Engineering features...")

def extract_features(df):
    # === BASIC TEXT STATS ===
    df["char_count"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["sentence_count"] = df["text"].apply(lambda x: max(1, len(re.split(r'[.!?]', x))))
    df["avg_word_len"] = df["text"].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
    df["unique_words"] = df["text"].apply(lambda x: len(set(x.split())))
    df["lexical_diversity"] = df["unique_words"] / df["word_count"].replace(0, 1)
    
    # === NUMBERS & CONSTRAINTS ===
    def extract_numbers(text):
        nums = [int(n) for n in re.findall(r'\b\d+\b', text) if len(n) <= 10]  # Avoid huge numbers
        if not nums:
            return 0, 0, 0
        return len(nums), max(nums), np.mean(nums)
    
    df["num_count"], df["max_num"], df["avg_num"] = zip(*df["text"].apply(extract_numbers))
    df["log_max_num"] = df["max_num"].apply(lambda x: math.log10(x + 1))
    df["has_10k"] = (df["max_num"] >= 10000).astype(int)
    df["has_100k"] = (df["max_num"] >= 100000).astype(int)
    df["has_1m"] = (df["max_num"] >= 1000000).astype(int)
    
    # === SYMBOLS & SPECIAL CHARS ===
    df["math_ops"] = df["text"].apply(lambda x: x.count('+') + x.count('-') + x.count('*') + x.count('/'))
    df["comparison_ops"] = df["text"].apply(lambda x: x.count('<') + x.count('>') + x.count('='))
    df["brackets"] = df["text"].apply(lambda x: x.count('(') + x.count('[') + x.count('{'))
    
    # === DIFFICULTY KEYWORDS (WEIGHTED) ===
    # Hard indicators
    hard_kw = [
        r'dynamic programming', r'\bdp\b', r'memoization', r'tabulation',
        r'dijkstra', r'bellman', r'floyd', r'kruskal', r'prim',
        r'segment tree', r'fenwick', r'trie', r'suffix',
        r'topological', r'strongly connected', r'articulation',
        r'nondeterministic', r'np-hard', r'exponential',
        r'optimize', r'optimal substructure', r'minimize complexity'
    ]
    df["hard_keywords"] = df["text"].apply(lambda x: sum(1 for kw in hard_kw if re.search(kw, x)))
    
    # Medium indicators  
    med_kw = [
        r'binary search', r'two pointer', r'sliding window',
        r'greedy', r'backtrack', r'dfs', r'bfs',
        r'graph', r'tree', r'heap', r'priority queue',
        r'sort', r'merge sort', r'quick sort',
        r'hash', r'map', r'dictionary',
        r'prefix sum', r'difference array',
        r'recursion', r'divide and conquer'
    ]
    df["medium_keywords"] = df["text"].apply(lambda x: sum(1 for kw in med_kw if re.search(kw, x)))
    
    # Easy indicators
    easy_kw = [
        r'print', r'output', r'read input', r'sum', r'count',
        r'reverse', r'palindrome check', r'even or odd',
        r'simple', r'basic', r'straightforward',
        r'array manipulation', r'string manipulation'
    ]
    df["easy_keywords"] = df["text"].apply(lambda x: sum(1 for kw in easy_kw if re.search(kw, x)))
    
    # Keyword ratios (more discriminative)
    total_kw = df["hard_keywords"] + df["medium_keywords"] + df["easy_keywords"] + 0.1
    df["hard_ratio"] = df["hard_keywords"] / total_kw
    df["medium_ratio"] = df["medium_keywords"] / total_kw
    df["easy_ratio"] = df["easy_keywords"] / total_kw
    
    # === PROBLEM STRUCTURE ===
    df["multi_testcase"] = df["text"].str.contains(r'test case|multiple case|\bt\b test').astype(int)
    df["single_output"] = df["text"].str.contains(r'single line|one line output').astype(int)
    df["matrix_grid"] = df["text"].str.contains(r'matrix|grid|2d array|table').astype(int)
    df["graph_problem"] = df["text"].str.contains(r'graph|tree|node|edge|vertex').astype(int)
    df["string_problem"] = df["text"].str.contains(r'string|character|substring|word').astype(int)
    df["array_problem"] = df["text"].str.contains(r'array|list|sequence').astype(int)
    
    # === PROBLEM TYPE ===
    df["asks_optimal"] = df["text"].str.contains(r'minimum|maximum|optimal|best|least|most').astype(int)
    df["asks_count"] = df["text"].str.contains(r'how many|count|number of').astype(int)
    df["asks_existence"] = df["text"].str.contains(r'possible|can you|is it possible|exists').astype(int)
    df["asks_output"] = df["text"].str.contains(r'print|output|display|return').astype(int)
    
    # === CONSTRAINTS ===
    df["has_constraints"] = df["text"].str.contains(r'constraint|limit|bound').astype(int)
    df["time_mentioned"] = df["text"].str.contains(r'time limit|time complexity|efficient').astype(int)
    df["space_mentioned"] = df["text"].str.contains(r'space|memory').astype(int)
    
    # === ADVANCED FEATURES ===
    # Words per sentence (complex problems have longer explanations)
    df["words_per_sentence"] = df["word_count"] / df["sentence_count"]
    
    # Technical vocabulary (long, complex words)
    df["tech_words"] = df["text"].apply(lambda x: sum(1 for w in x.split() if len(w) > 12))
    df["tech_ratio"] = df["tech_words"] / df["word_count"].replace(0, 1)
    
    # Question complexity (multiple questions = harder)
    df["question_marks"] = df["text"].apply(lambda x: x.count('?'))
    
    return df

df = extract_features(df)
print(f"  ✓ Created {len([c for c in df.columns if c not in ['title', 'description', 'input_description', 'output_description', 'text', 'problem_class', 'problem_score']])} features")

# ======================
# 4. PREPARE TARGETS
# ======================
print("\n[4/7] Preparing targets...")
le = LabelEncoder()
df["class_label"] = le.fit_transform(df["problem_class"])
y_class = df["class_label"]
y_reg = df["problem_score"]
print(f"  ✓ Classes: {le.classes_}")

# ======================
# 5. TF-IDF + FEATURE COMBINATION
# ======================
print("\n[5/7] Creating TF-IDF features...")

# Two TF-IDF vectorizers: one for words, one for character n-grams
tfidf_word = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.8,
    stop_words='english',
    sublinear_tf=True
)

tfidf_char = TfidfVectorizer(
    max_features=1000,
    analyzer='char',
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True
)

X_word = tfidf_word.fit_transform(df["text"])
X_char = tfidf_char.fit_transform(df["text"])
print(f"  ✓ Word TF-IDF: {X_word.shape[1]} features")
print(f"  ✓ Char TF-IDF: {X_char.shape[1]} features")

# Manual features
feature_cols = [
    "char_count", "word_count", "sentence_count", "avg_word_len",
    "unique_words", "lexical_diversity", "words_per_sentence",
    "num_count", "max_num", "avg_num", "log_max_num",
    "has_10k", "has_100k", "has_1m",
    "math_ops", "comparison_ops", "brackets",
    "hard_keywords", "medium_keywords", "easy_keywords",
    "hard_ratio", "medium_ratio", "easy_ratio",
    "multi_testcase", "single_output", "matrix_grid",
    "graph_problem", "string_problem", "array_problem",
    "asks_optimal", "asks_count", "asks_existence", "asks_output",
    "has_constraints", "time_mentioned", "space_mentioned",
    "tech_words", "tech_ratio", "question_marks"
]

scaler = StandardScaler()
X_manual = scaler.fit_transform(df[feature_cols])

# Combine all features
X = hstack([X_word, X_char, X_manual])
print(f"  ✓ Total features: {X.shape[1]}")

# ======================
# 6. TRAIN-TEST SPLIT + SMOTE
# ======================
print("\n[6/7] Splitting data and balancing...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print(f"  Before SMOTE: {len(y_train)} samples")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE:  {len(y_train_bal)} samples (balanced)")

# ======================
# 7. TRAIN MODELS
# ======================
print("\n[7/7] Training models...")

# Model 1: Gradient Boosting
print("  [1/3] Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)
gb.fit(X_train_bal, y_train_bal)
gb_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"        Accuracy: {gb_acc:.4f}")

# Model 2: Random Forest
print("  [2/3] Random Forest...")
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_bal, y_train_bal)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"        Accuracy: {rf_acc:.4f}")

# Model 3: Logistic Regression (L2 regularized)
print("  [3/3] Logistic Regression...")
lr = LogisticRegression(
    C=0.5,
    max_iter=1000,
    class_weight='balanced',
    solver='saga',
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train_bal, y_train_bal)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"        Accuracy: {lr_acc:.4f}")

# Ensemble
print("\n  Creating ensemble...")
gb_proba = gb.predict_proba(X_test)
rf_proba = rf.predict_proba(X_test)
lr_proba = lr.predict_proba(X_test)

weights = np.array([gb_acc**2, rf_acc**2, lr_acc**2])
weights = weights / weights.sum()

ensemble_proba = weights[0]*gb_proba + weights[1]*rf_proba + weights[2]*lr_proba
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"  Ensemble accuracy: {ensemble_acc:.4f}")

# Choose best
best = max([
    ('Gradient Boosting', gb, gb_acc, gb_pred),
    ('Random Forest', rf, rf_acc, rf_pred),
    ('Logistic Regression', lr, lr_acc, lr_pred),
    ('Ensemble', gb, ensemble_acc, ensemble_pred)
], key=lambda x: x[2])

clf = best[1]
final_acc = best[2]
final_pred = best[3]

print(f"\n  ✓ Best model: {best[0]} ({final_acc:.4f})")

# ======================
# EVALUATION
# ======================
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\n🎯 Accuracy: {final_acc:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, final_pred, target_names=le.classes_, digits=4))

print("📈 Confusion Matrix:")
cm = confusion_matrix(y_test, final_pred)
print(cm)

print("\n✅ Per-Class Accuracy:")
for i, cls in enumerate(le.classes_):
    recall = cm[i,i] / cm[i].sum()
    print(f"  {cls.capitalize():8s}: {recall:.2%} ({cm[i,i]}/{cm[i].sum()})")

# Feature Importance
print("\n🔍 Top 20 Features:")
if hasattr(clf, 'feature_importances_'):
    word_feats = list(tfidf_word.get_feature_names_out())
    char_feats = [f"char_{i}" for i in range(X_char.shape[1])]
    all_feats = word_feats + char_feats + feature_cols
    
    imp = clf.feature_importances_
    top_idx = np.argsort(imp)[-20:][::-1]
    
    for i, idx in enumerate(top_idx, 1):
        name = all_feats[idx] if idx < len(all_feats) else f"feat_{idx}"
        print(f"  {i:2d}. {name[:35]:35s} {imp[idx]:.5f}")

# ======================
# REGRESSION MODEL
# ======================
print("\n" + "="*60)
print("REGRESSION MODEL")
print("="*60)

X_tr, X_te, y_tr, y_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
reg.fit(X_tr, y_tr)
pred_reg = reg.predict(X_te)

mae = mean_absolute_error(y_te, pred_reg)
rmse = np.sqrt(mean_squared_error(y_te, pred_reg))

print(f"\n  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")

# ======================
# SAVE MODELS
# ======================
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

joblib.dump(clf, "classifier.pkl")
joblib.dump(reg, "regressor.pkl")
joblib.dump(tfidf_word, "tfidf_word.pkl")
joblib.dump(tfidf_char, "tfidf_char.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

if ensemble_acc >= max(gb_acc, rf_acc, lr_acc):
    joblib.dump([gb, rf, lr], "ensemble_models.pkl")
    joblib.dump(weights, "ensemble_weights.pkl")

print("\n✅ Models saved:")
print("  • classifier.pkl")
print("  • regressor.pkl")
print("  • tfidf_word.pkl, tfidf_char.pkl")
print("  • scaler.pkl, label_encoder.pkl")
print("  • feature_cols.pkl")

print("\n" + "="*60)
print("✨ TRAINING COMPLETE!")
print("="*60)
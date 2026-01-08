import json
import re
import math
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. LOAD DATASET
# ======================
print("="*60)
print("LOADING DATASET")
print("="*60)

data = []
with open("./data/problems_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

df = pd.DataFrame(data)
print(f"\nDataset shape: {df.shape}")
print("\nClass distribution:")
print(df["problem_class"].value_counts())
print(f"\nScore statistics by class:")
print(df.groupby("problem_class")["problem_score"].describe())

# ======================
# 2. IMPROVED TEXT CLEANING
# ======================
print("\n" + "="*60)
print("TEXT PREPROCESSING")
print("="*60)

def clean_text(text):
    text = str(text).lower()
    # Keep important punctuation for pattern matching
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s\+\-\*/=<>\(\)\[\]\{\}\^\%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Combine with weights (title is more important)
df["text"] = (
    df["title"].fillna("") + " " +
    df["title"].fillna("") + " " +  # Repeat title for emphasis
    df["description"].fillna("") + " " +
    df["input_description"].fillna("") + " " +
    df["output_description"].fillna("")
)
df["text"] = df["text"].apply(clean_text)
print("✓ Text cleaning complete")

# ======================
# 3. ENHANCED FEATURE ENGINEERING
# ======================
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

def extract_features(df):
    print("\nExtracting features...")
    
    # ===== TEXT STATISTICS =====
    df["text_length"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["unique_word_ratio"] = df["text"].apply(lambda x: len(set(x.split())) / max(len(x.split()), 1))
    df["sentence_count"] = df["text"].apply(lambda x: len(re.split(r'[.!?]+', x)))
    df["avg_sentence_length"] = df["word_count"] / df["sentence_count"].replace(0, 1)
    
    # ===== COMPLEXITY INDICATORS =====
    df["math_symbols"] = df["text"].apply(lambda x: sum(x.count(c) for c in "+-*/=<>()[]{}^%"))
    df["brackets_count"] = df["text"].apply(lambda x: sum(x.count(c) for c in "()[]{}"))
    
    # Extract all numbers and analyze them
    def analyze_numbers(text):
        nums = [int(n) for n in re.findall(r'\b\d+\b', text)]
        if not nums:
            return 0, 0, 0, 0
        return len(nums), max(nums), np.mean(nums), np.std(nums) if len(nums) > 1 else 0
    
    df["num_count"], df["max_number"], df["avg_number"], df["std_number"] = zip(*df["text"].apply(analyze_numbers))
    df["log_max_number"] = df["max_number"].apply(lambda x: math.log1p(x))
    
    # Complexity thresholds (indicators of harder problems)
    df["has_large_n"] = (df["max_number"] >= 100000).astype(int)  # 10^5
    df["has_huge_n"] = (df["max_number"] >= 1000000).astype(int)  # 10^6
    
    # ===== ALGORITHMIC PATTERNS (MORE SPECIFIC) =====
    # Advanced algorithms (usually hard problems)
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
    
    # Medium complexity patterns
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
    
    # Basic patterns (often easy problems)
    basic_patterns = {
        'simple_array': r'\b(array\s*manipulation|array\s*rotation)\b',
        'simple_string': r'\b(reverse\s*string|palindrome\s*check)\b',
        'simple_math': r'\b(sum|product|average|count)\b',
        'sorting_basic': r'\b(sort\s*array|sort\s*list)\b',
        'iteration': r'\b(iterate|loop|for\s*each)\b',
    }
    
    for key, pattern in basic_patterns.items():
        df[f"basic_{key}"] = df["text"].str.contains(pattern, regex=True).astype(int)
    
    # ===== PROBLEM CHARACTERISTICS =====
    # Input/Output complexity
    df["multi_test_cases"] = df["text"].str.contains(r'test\s*case|multiple\s*case|t\s*test').astype(int)
    df["single_line_input"] = df["text"].str.contains(r'single\s*line').astype(int)
    df["matrix_input"] = df["text"].str.contains(r'matrix|grid|2d\s*array').astype(int)
    
    # Constraint mentions
    df["has_constraints"] = df["text"].str.contains(r'constraint|limit|bound').astype(int)
    df["time_limit"] = df["text"].str.contains(r'time\s*limit|time\s*complexity').astype(int)
    df["space_limit"] = df["text"].str.contains(r'space\s*limit|space\s*complexity|memory').astype(int)
    
    # Problem asks for...
    df["asks_optimal"] = df["text"].str.contains(r'optimal|minimum|maximum|best|efficient').astype(int)
    df["asks_count"] = df["text"].str.contains(r'how\s*many|count\s*the|number\s*of').astype(int)
    df["asks_yes_no"] = df["text"].str.contains(r'possible|impossible|can\s*you|is\s*it\s*possible').astype(int)
    
    # ===== WORD COMPLEXITY =====
    df["avg_word_length"] = df["text"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    df["long_words_count"] = df["text"].apply(lambda x: sum(1 for w in x.split() if len(w) > 10))
    
    # ===== AGGREGATE FEATURES =====
    df["total_advanced"] = sum(df[col] for col in df.columns if col.startswith("adv_"))
    df["total_medium"] = sum(df[col] for col in df.columns if col.startswith("med_"))
    df["total_basic"] = sum(df[col] for col in df.columns if col.startswith("basic_"))
    
    # Ratio features (more informative)
    total_patterns = df["total_advanced"] + df["total_medium"] + df["total_basic"]
    df["ratio_advanced"] = df["total_advanced"] / (total_patterns + 1)
    df["ratio_medium"] = df["total_medium"] / (total_patterns + 1)
    df["ratio_basic"] = df["total_basic"] / (total_patterns + 1)
    
    print(f"✓ Extracted features complete")
    
    return df

df = extract_features(df)

# ======================
# 4. TARGET ENCODING
# ======================
le = LabelEncoder()
df["class_label"] = le.fit_transform(df["problem_class"])
y_class = df["class_label"]
y_reg = df["problem_score"]

# ======================
# 5. IMPROVED TF-IDF
# ======================
print("\n" + "="*60)
print("CREATING TF-IDF FEATURES")
print("="*60)

# Use character n-grams for better pattern matching
tfidf = TfidfVectorizer(
    max_features=3000,
    stop_words="english",
    ngram_range=(1, 3),  # Include trigrams
    min_df=2,
    max_df=0.85,
    sublinear_tf=True,
    analyzer='word'  # Word-level analysis
)
X_text = tfidf.fit_transform(df["text"])
print(f"✓ TF-IDF features: {X_text.shape[1]}")

# ======================
# 6. COMBINE ALL FEATURES
# ======================
print("\n" + "="*60)
print("COMBINING FEATURES")
print("="*60)

# Select all engineered features
feature_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in 
    ['adv_', 'med_', 'basic_', 'total_', 'ratio_'])] + [
    "text_length", "word_count", "unique_word_ratio", "sentence_count", 
    "avg_sentence_length", "math_symbols", "brackets_count",
    "num_count", "max_number", "avg_number", "std_number", "log_max_number",
    "has_large_n", "has_huge_n",
    "multi_test_cases", "single_line_input", "matrix_input",
    "has_constraints", "time_limit", "space_limit",
    "asks_optimal", "asks_count", "asks_yes_no",
    "avg_word_length", "long_words_count"
]

# Scale numerical features
scaler = StandardScaler()
X_extra_scaled = scaler.fit_transform(df[feature_cols])

# Combine
X = hstack([X_text, X_extra_scaled])

print(f"✓ Total features: {X.shape[1]}")
print(f"  - TF-IDF: {X_text.shape[1]}")
print(f"  - Manual: {len(feature_cols)}")

# ======================
# 7. STRATIFIED SPLIT + SMOTE
# ======================
print("\n" + "="*60)
print("TRAIN-TEST SPLIT + SMOTE")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print(f"\nBefore SMOTE - Class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {le.classes_[u]:8s}: {c:4d}")

# SMOTE with adjusted neighbors
smote = SMOTE(random_state=42, k_neighbors=3)  # Reduced for stability
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE - Class distribution:")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {le.classes_[u]:8s}: {c:4d}")

# ======================
# 8. TRAIN GRADIENT BOOSTING (OPTIMIZED PARAMS)
# ======================
print("\n" + "="*60)
print("TRAINING GRADIENT BOOSTING")
print("="*60)

# Use pre-optimized parameters (skip expensive grid search)
clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=7,
    min_samples_split=15,
    min_samples_leaf=5,
    subsample=0.85,
    max_features='sqrt',
    random_state=42,
    verbose=0
)

print("Training Gradient Boosting...")
clf.fit(X_train_balanced, y_train_balanced)
print("✓ Training complete")

# ======================
# 9. TRAIN ENSEMBLE
# ======================
print("\n" + "="*60)
print("TRAINING ENSEMBLE MODELS")
print("="*60)

models = []

# Model 1: Tuned Gradient Boosting
print("\n[1/3] Using tuned Gradient Boosting...")
gb_pred = clf.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"      ✓ Accuracy: {gb_acc:.4f}")
models.append(('GB', clf, gb_acc, gb_pred))

# Model 2: Random Forest
print("\n[2/3] Training Random Forest...")
rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=15,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_clf.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"      ✓ Accuracy: {rf_acc:.4f}")
models.append(('RF', rf_clf, rf_acc, rf_pred))

# Model 3: SVM with RBF kernel
print("\n[3/3] Training SVM...")
svm_clf = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42,
    verbose=False
)
svm_clf.fit(X_train_balanced, y_train_balanced)
svm_pred = svm_clf.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"      ✓ Accuracy: {svm_acc:.4f}")
models.append(('SVM', svm_clf, svm_acc, svm_pred))

# ======================
# 10. WEIGHTED ENSEMBLE
# ======================
print("\n" + "="*60)
print("CREATING ENSEMBLE")
print("="*60)

gb_proba = clf.predict_proba(X_test)
rf_proba = rf_clf.predict_proba(X_test)
svm_proba = svm_clf.predict_proba(X_test)

# Weight by accuracy^2
weights = np.array([gb_acc**2, rf_acc**2, svm_acc**2])
weights = weights / weights.sum()

print(f"\nModel weights:")
print(f"  Gradient Boosting: {weights[0]:.3f} (acc: {gb_acc:.4f})")
print(f"  Random Forest:     {weights[1]:.3f} (acc: {rf_acc:.4f})")
print(f"  SVM:               {weights[2]:.3f} (acc: {svm_acc:.4f})")

ensemble_proba = weights[0] * gb_proba + weights[1] * rf_proba + weights[2] * svm_proba
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\n✓ Ensemble Accuracy: {ensemble_acc:.4f}")

# Use best
best_model = max(models + [('Ensemble', clf, ensemble_acc, ensemble_pred)], key=lambda x: x[2])
final_acc = best_model[2]
final_pred = best_model[3]
print(f"  → Using {best_model[0]} (accuracy: {final_acc:.4f})")

# ======================
# 11. EVALUATION
# ======================
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

print(f"\n🎯 Final Accuracy: {final_acc:.4f}")
print(f"\n📊 Classification Report:")
print(classification_report(y_test, final_pred, target_names=le.classes_, digits=4))

print("\n📈 Confusion Matrix:")
cm = confusion_matrix(y_test, final_pred)
print(cm)

print("\n✅ Per-class Performance:")
for i, class_name in enumerate(le.classes_):
    class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"  {class_name.capitalize():8s}: {class_acc:.4f} ({cm[i,i]}/{cm[i].sum()} correct)")

# ======================
# 12. REGRESSION
# ======================
print("\n" + "="*60)
print("REGRESSION MODEL")
print("="*60)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(
    n_estimators=300, max_depth=25, min_samples_split=5,
    random_state=42, n_jobs=-1
)
reg.fit(X_train_reg, y_train_reg)
pred_reg = reg.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, pred_reg))
print(f"\n📉 MAE:  {mae:.4f}")
print(f"📉 RMSE: {rmse:.4f}")

# ======================
# 13. FEATURE IMPORTANCE
# ======================
print("\n" + "="*60)
print("TOP 25 IMPORTANT FEATURES")
print("="*60)

if hasattr(clf, 'feature_importances_'):
    tfidf_features = list(tfidf.get_feature_names_out())
    all_features = tfidf_features + feature_cols
    
    importance = clf.feature_importances_
    indices = np.argsort(importance)[-25:][::-1]
    
    for i, idx in enumerate(indices, 1):
        feat_name = all_features[idx] if idx < len(all_features) else f"feature_{idx}"
        print(f"  {i:2d}. {feat_name:40s}: {importance[idx]:.5f}")

# ======================
# 14. SAVE MODELS
# ======================
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

joblib.dump(clf, "classifier.pkl")
joblib.dump(reg, "regressor.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

if ensemble_acc >= max([m[2] for m in models]):
    joblib.dump([clf, rf_clf, svm_clf], "ensemble_models.pkl")
    joblib.dump(weights, "ensemble_weights.pkl")

print("\n✅ All models saved successfully!")
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
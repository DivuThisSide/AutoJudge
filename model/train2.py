import json
import re
import math
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
# 2. TEXT CLEANING
# ======================
print("\n" + "="*60)
print("TEXT PREPROCESSING")
print("="*60)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    # Keep more special chars that indicate complexity
    text = re.sub(r"[^a-z0-9\s\+\-\*/=<>\(\)\[\]\{\}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Combine text fields
df["text"] = (
    df["title"].fillna("") + " " +
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
    
    # Basic text features
    df["text_length"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["sentence_count"] = df["text"].apply(lambda x: len(re.split(r'[.!?]+', x)))
    df["math_symbols"] = df["text"].apply(
        lambda x: sum(x.count(c) for c in "+-*/=<>()[]{}"))
    
    # Algorithm/DS keywords (comprehensive)
    keywords = {
        # Advanced algorithms
        'dp': r'\b(dp|dynamic\s*programming|memoization|tabulation|optimal\s*substructure)\b',
        'graph_advanced': r'\b(dijkstra|bellman|floyd|kruskal|prim|tarjan|articulation|strongly\s*connected|topological)\b',
        'graph_basic': r'\b(graph|tree|node|edge|vertex|bfs|dfs|traversal|path|cycle)\b',
        'segment_tree': r'\b(segment\s*tree|fenwick|bit|binary\s*indexed)\b',
        'trie': r'\b(trie|prefix\s*tree)\b',
        'union_find': r'\b(union\s*find|disjoint\s*set|dsu)\b',
        
        # Medium complexity
        'greedy': r'\b(greedy|optimal|maximize|minimize|locally\s*optimal)\b',
        'binary_search': r'\b(binary\s*search|bisect|logarithmic|lower\s*bound|upper\s*bound)\b',
        'two_pointer': r'\b(two\s*pointer|sliding\s*window)\b',
        'backtrack': r'\b(backtrack|permutation|combination|subset|pruning)\b',
        'divide_conquer': r'\b(divide\s*and\s*conquer|merge\s*sort)\b',
        
        # Data structures
        'string_pattern': r'\b(substring|palindrome|pattern|kmp|rabin\s*karp|suffix)\b',
        'sorting': r'\b(sort|sorted|order|arrange|quicksort)\b',
        'array': r'\b(array|list|sequence|subarray)\b',
        'matrix': r'\b(matrix|grid|2d\s*array)\b',
        'stack_queue': r'\b(stack|queue|deque|priority\s*queue|heap|monotonic)\b',
        'linked_list': r'\b(linked\s*list|node\s*next|pointer)\b',
        'hash': r'\b(hash|map|dictionary|set)\b',
        
        # Mathematical concepts
        'math': r'\b(prime|gcd|lcm|modulo|factorial|fibonacci|combinatorics)\b',
        'geometry': r'\b(geometry|point|line|angle|distance|coordinate)\b',
        'number_theory': r'\b(number\s*theory|divisor|factor|coprime)\b',
        
        # General CS concepts
        'recursion': r'\b(recursion|recursive)\b',
        'bit_manipulation': r'\b(bit|bitwise|xor|and|or|shift)\b',
        'simulation': r'\b(simulate|simulation|iteration)\b',
    }
    
    for key, pattern in keywords.items():
        df[f"kw_{key}"] = df["text"].str.contains(pattern, regex=True).astype(int)
    
    # Count total algorithm mentions
    df["total_algo_keywords"] = sum(df[col] for col in df.columns if col.startswith("kw_"))
    
    # Complexity indicators from constraints
    df["has_constraints"] = df["text"].str.contains(r'constraint|limit|bound').astype(int)
    df["num_numbers"] = df["text"].apply(lambda x: len(re.findall(r'\b\d+\b', x)))
    
    # Extract largest number (complexity indicator)
    def get_max_number(text):
        nums = re.findall(r'\b\d+\b', text)
        return max([int(n) for n in nums]) if nums else 0
    
    df["max_constraint"] = df["text"].apply(get_max_number)
    df["log_max_constraint"] = df["max_constraint"].apply(lambda x: math.log1p(x))
    
    # Large constraint indicator (10^5 or more suggests harder problem)
    df["has_large_constraint"] = (df["max_constraint"] >= 100000).astype(int)
    
    # Word complexity
    df["avg_word_length"] = df["text"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    df["long_words_ratio"] = df["text"].apply(
        lambda x: sum(1 for w in x.split() if len(w) > 8) / max(len(x.split()), 1)
    )
    
    # Special characters ratio
    df["special_char_ratio"] = df["text"].apply(
        lambda x: sum(c in "+-*/=<>()[]{}^%&|!" for c in x) / max(len(x), 1)
    )
    
    # Complexity mentions
    df["has_time_complexity"] = df["text"].str.contains(r'time\s*complexity|o\(|runtime').astype(int)
    df["has_space_complexity"] = df["text"].str.contains(r'space\s*complexity|memory').astype(int)
    df["has_optimization"] = df["text"].str.contains(r'optim|efficien|fast').astype(int)
    
    # Test case indicators (more test cases = often harder)
    df["mentions_testcase"] = df["text"].str.contains(r'test\s*case|example|sample').astype(int)
    
    # Multiple solutions indicator
    df["mentions_multiple_sol"] = df["text"].str.contains(r'multiple\s*way|various\s*approach|different\s*method').astype(int)
    
    print(f"✓ Extracted {len([col for col in df.columns if col.startswith('kw_')])} keyword features")
    print(f"✓ Total manual features: {len([col for col in df.columns if col.startswith('kw_') or col in ['text_length', 'word_count']])}")
    
    return df

df = extract_features(df)

# ======================
# 4. TARGET ENCODING (BEFORE SCORE FEATURES!)
# ======================
le = LabelEncoder()
df["class_label"] = le.fit_transform(df["problem_class"])
y_class = df["class_label"]
y_reg = df["problem_score"]

# ======================
# 5. ADD SCORE FEATURES (ONLY FOR COMPARISON - WILL BE REMOVED)
# ======================
print("\n" + "="*60)
print("SCORE-BASED FEATURES (CHECKING DATA LEAKAGE)")
print("="*60)

# Create score features temporarily to show the leakage problem
df["score_normalized"] = df["problem_score"].apply(lambda x: (x - df["problem_score"].min()) / (df["problem_score"].max() - df["problem_score"].min()))
q33 = df["problem_score"].quantile(0.33)
q67 = df["problem_score"].quantile(0.67)
df["score_bin_low"] = df["problem_score"].apply(lambda x: 1 if x < q33 else 0)
df["score_bin_high"] = df["problem_score"].apply(lambda x: 1 if x > q67 else 0)

print(f"✓ Score range: {df['problem_score'].min():.2f} to {df['problem_score'].max():.2f}")
print(f"⚠️  WARNING: Score features will NOT be used in final model (data leakage)")
print(f"⚠️  These scores would only be available AFTER solving the problem!")

# ======================
# 6. TF-IDF FEATURES
# ======================
print("\n" + "="*60)
print("CREATING TF-IDF FEATURES")
print("="*60)

tfidf = TfidfVectorizer(
    max_features=2500,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.7,
    sublinear_tf=True
)
X_text = tfidf.fit_transform(df["text"])
print(f"✓ TF-IDF features: {X_text.shape[1]}")

# ======================
# 7. COMBINE ALL FEATURES (WITHOUT SCORE - NO DATA LEAKAGE!)
# ======================
print("\n" + "="*60)
print("COMBINING FEATURES")
print("="*60)

# DO NOT include score features - they cause data leakage!
feature_cols = (
    [col for col in df.columns if col.startswith("kw_")] + 
    ["text_length", "word_count", "sentence_count", "math_symbols",
     "total_algo_keywords", "has_constraints", "num_numbers",
     "max_constraint", "log_max_constraint", "has_large_constraint",
     "avg_word_length", "long_words_ratio", "special_char_ratio",
     "has_time_complexity", "has_space_complexity", "has_optimization",
     "mentions_testcase", "mentions_multiple_sol"]
    # REMOVED: "score_normalized", "score_bin_low", "score_bin_high"
)

# Scale numerical features
scaler = StandardScaler()
X_extra_scaled = scaler.fit_transform(df[feature_cols])

# Combine TF-IDF + manual features
X = hstack([X_text, X_extra_scaled])

print(f"✓ Total features: {X.shape[1]}")
print(f"  - TF-IDF: {X_text.shape[1]}")
print(f"  - Manual: {len(feature_cols)}")

# ======================
# 8. TRAIN-TEST SPLIT WITH SMOTE
# ======================
print("\n" + "="*60)
print("TRAIN-TEST SPLIT + SMOTE BALANCING")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print(f"\nBefore SMOTE:")
print(f"  Train size: {X_train.shape[0]}")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"    {le.classes_[u]:8s}: {c}")

# Apply SMOTE to balance training data
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"  Train size: {X_train_balanced.shape[0]}")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for u, c in zip(unique, counts):
    print(f"    {le.classes_[u]:8s}: {c}")

# ======================
# 9. TRAIN MULTIPLE MODELS
# ======================
print("\n" + "="*60)
print("TRAINING CLASSIFICATION MODELS")
print("="*60)

models_to_train = []

# Model 1: Gradient Boosting
print("\n[1/3] Training Gradient Boosting...")
gb_clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=8,
    min_samples_split=15,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    random_state=42,
    verbose=0
)
gb_clf.fit(X_train_balanced, y_train_balanced)
gb_pred = gb_clf.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"      ✓ Gradient Boosting Accuracy: {gb_acc:.4f}")
models_to_train.append(('GB', gb_clf, gb_acc, gb_pred))

# Model 2: Random Forest
print("\n[2/3] Training Random Forest...")
rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_clf.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"      ✓ Random Forest Accuracy: {rf_acc:.4f}")
models_to_train.append(('RF', rf_clf, rf_acc, rf_pred))

# Model 3: Logistic Regression
print("\n[3/3] Training Logistic Regression...")
lr_clf = LogisticRegression(
    max_iter=1000,
    C=0.3,
    class_weight='balanced',
    solver='saga',
    penalty='l2',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
lr_clf.fit(X_train_balanced, y_train_balanced)
lr_pred = lr_clf.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"      ✓ Logistic Regression Accuracy: {lr_acc:.4f}")
models_to_train.append(('LR', lr_clf, lr_acc, lr_pred))

# ======================
# 10. ENSEMBLE WITH WEIGHTED VOTING
# ======================
print("\n" + "="*60)
print("CREATING ENSEMBLE")
print("="*60)

# Get probabilities from all models
gb_proba = gb_clf.predict_proba(X_test)
rf_proba = rf_clf.predict_proba(X_test)
lr_proba = lr_clf.predict_proba(X_test)

# Weight by individual accuracy (squared for more emphasis on better models)
weights = np.array([gb_acc**2, rf_acc**2, lr_acc**2])
weights = weights / weights.sum()

print(f"\nModel weights:")
print(f"  Gradient Boosting: {weights[0]:.3f}")
print(f"  Random Forest:     {weights[1]:.3f}")
print(f"  Logistic Reg:      {weights[2]:.3f}")

# Weighted ensemble
ensemble_proba = (
    weights[0] * gb_proba +
    weights[1] * rf_proba +
    weights[2] * lr_proba
)
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\n✓ Ensemble Accuracy: {ensemble_acc:.4f}")

# Choose best approach
if ensemble_acc >= max(gb_acc, rf_acc, lr_acc):
    print("  → Using ensemble predictions")
    final_pred = ensemble_pred
    final_acc = ensemble_acc
    clf = gb_clf  # Save best model for feature importance
else:
    best_model = max(models_to_train, key=lambda x: x[2])
    clf = best_model[1]
    final_acc = best_model[2]
    final_pred = best_model[3]
    print(f"  → Using {best_model[0]} (best single model)")

# ======================
# 11. DETAILED EVALUATION
# ======================
print("\n" + "="*60)
print("FINAL CLASSIFICATION RESULTS")
print("="*60)

print(f"\n🎯 Final Accuracy: {final_acc:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, final_pred, target_names=le.classes_, digits=4))

print("\n📈 Confusion Matrix:")
cm = confusion_matrix(y_test, final_pred)
print(cm)

print("\n✅ Per-class Accuracy:")
for i, class_name in enumerate(le.classes_):
    class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    class_total = cm[i].sum()
    print(f"  {class_name.capitalize():8s}: {class_acc:.4f} ({cm[i,i]}/{class_total} correct)")

# ======================
# 12. REGRESSION MODEL
# ======================
print("\n" + "="*60)
print("TRAINING REGRESSION MODEL")
print("="*60)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

reg.fit(X_train_reg, y_train_reg)
pred_reg = reg.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, pred_reg))

print(f"\n📉 Regression Results:")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")

# ======================
# 13. FEATURE IMPORTANCE
# ======================
print("\n" + "="*60)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*60)

if hasattr(clf, 'feature_importances_'):
    tfidf_features = tfidf.get_feature_names_out()
    all_features = list(tfidf_features) + feature_cols
    
    importance = clf.feature_importances_
    indices = np.argsort(importance)[-20:][::-1]
    
    for i, idx in enumerate(indices, 1):
        feat_name = all_features[idx] if idx < len(all_features) else f"feature_{idx}"
        print(f"  {i:2d}. {feat_name:35s} : {importance[idx]:.5f}")

# ======================
# 14. SAVE ALL MODELS & ARTIFACTS
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

# Save ensemble models if used
if ensemble_acc >= max(gb_acc, rf_acc, lr_acc):
    joblib.dump([gb_clf, rf_clf, lr_clf], "ensemble_models.pkl")
    joblib.dump(weights, "ensemble_weights.pkl")

print("\n✅ Models saved successfully!")
print("\nSaved files:")
print("  • classifier.pkl")
print("  • regressor.pkl")
print("  • tfidf.pkl")
print("  • label_encoder.pkl")
print("  • scaler.pkl")
print("  • feature_cols.pkl")
if ensemble_acc >= max(gb_acc, rf_acc, lr_acc):
    print("  • ensemble_models.pkl")
    print("  • ensemble_weights.pkl")

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
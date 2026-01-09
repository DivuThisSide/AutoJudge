from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import re
import math
from scipy.sparse import hstack

app = Flask(__name__)
try:
    clf = joblib.load("classifier.pkl")
    # reg = joblib.load("regressor.pkl")
    tfidf = joblib.load("tfidf.pkl")
    le = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    # mean_train = joblib.load("mean_train.pkl")
    # mean_pred = joblib.load("mean_pred.pkl")
    regressors = joblib.load("regressors_by_class.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# TEXT CLEANING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s\+\-\*/=<>\(\)\[\]\{\}\^\%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



# FEATURE EXTRACTION (MUST MATCH train3.py)
def extract_features(text):
    """Extract features exactly as in train3.py"""
    
    features = {}
    words = text.split()
    features["text_length"] = len(text)
    features["word_count"] = len(words)
    features["unique_word_ratio"] = len(set(words)) / max(len(words), 1)
    features["sentence_count"] = max(len(re.split(r'[.!?]+', text)), 1)
    features["avg_sentence_length"] = features["word_count"] / features["sentence_count"]
    features["math_symbols"] = sum(text.count(c) for c in "+-*/=<>()[]{}^%")
    features["brackets_count"] = sum(text.count(c) for c in "()[]{}")
    nums = [int(n) for n in re.findall(r'\b\d+\b', text) if len(n) <= 10]
    features["num_count"] = len(nums)
    features["max_number"] = max(nums) if nums else 0
    features["avg_number"] = np.mean(nums) if nums else 0
    features["std_number"] = np.std(nums) if len(nums) > 1 else 0
    features["log_max_number"] = math.log1p(features["max_number"])
    features["has_large_n"] = int(features["max_number"] >= 100000)
    features["has_huge_n"] = int(features["max_number"] >= 1000000)
    

    # Advanced patterns
    adv_patterns = {
        'adv_dp_explicit': r'\b(dynamic\s*programming|dp\s*solution|dp\s*table|memoization|tabulation)\b',
        'adv_graph_shortest': r'\b(dijkstra|bellman|floyd|shortest\s*path)\b',
        'adv_graph_spanning': r'\b(kruskal|prim|spanning\s*tree|mst)\b',
        'adv_graph_flow': r'\b(max\s*flow|min\s*cut|ford\s*fulkerson)\b',
        'adv_advanced_ds': r'\b(segment\s*tree|fenwick|trie|suffix)\b',
        'adv_string_algo': r'\b(kmp|rabin\s*karp|z\s*algorithm|manacher)\b',
        'adv_number_theory': r'\b(modular\s*arithmetic|gcd|lcm|coprime|prime\s*factorization)\b',
    }
    
    # Medium patterns
    med_patterns = {
        'med_two_pointer': r'\b(two\s*pointer|sliding\s*window)\b',
        'med_binary_search': r'\b(binary\s*search|bisect)\b',
        'med_greedy': r'\b(greedy|locally\s*optimal)\b',
        'med_backtrack': r'\b(backtrack|permutation|combination)\b',
        'med_bfs_dfs': r'\b(bfs|dfs|breadth\s*first|depth\s*first)\b',
        'med_graph_basic': r'\b(graph|tree|node|edge|adjacent)\b',
        'med_sorting_complex': r'\b(merge\s*sort|quick\s*sort|heap\s*sort)\b',
    }
    
    # Basic patterns
    basic_patterns = {
        'basic_simple_array': r'\b(array\s*manipulation|array\s*rotation)\b',
        'basic_simple_string': r'\b(reverse\s*string|palindrome\s*check)\b',
        'basic_simple_math': r'\b(sum|product|average|count)\b',
        'basic_sorting_basic': r'\b(sort\s*array|sort\s*list)\b',
        'basic_iteration': r'\b(iterate|loop|for\s*each)\b',
    }
    for key, pattern in {**adv_patterns, **med_patterns, **basic_patterns}.items():
        features[key] = int(bool(re.search(pattern, text)))
    
    # PROBLEM CHARACTERISTICS
    features["multi_test_cases"] = int(bool(re.search(r'test\s*case|multiple\s*case|t\s*test', text)))
    features["single_line_input"] = int(bool(re.search(r'single\s*line', text)))
    features["matrix_input"] = int(bool(re.search(r'matrix|grid|2d\s*array', text)))
    features["has_constraints"] = int(bool(re.search(r'constraint|limit|bound', text)))
    features["time_limit"] = int(bool(re.search(r'time\s*limit|time\s*complexity', text)))
    features["space_limit"] = int(bool(re.search(r'space\s*limit|space\s*complexity|memory', text)))
    features["asks_optimal"] = int(bool(re.search(r'optimal|minimum|maximum|best|efficient', text)))
    features["asks_count"] = int(bool(re.search(r'how\s*many|count\s*the|number\s*of', text)))
    features["asks_yes_no"] = int(bool(re.search(r'possible|impossible|can\s*you|is\s*it\s*possible', text)))
    features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
    features["long_words_count"] = sum(1 for w in words if len(w) > 10)

    features["total_advanced"] = sum(features.get(k, 0) for k in adv_patterns.keys())
    features["total_medium"] = sum(features.get(k, 0) for k in med_patterns.keys())
    features["total_basic"] = sum(features.get(k, 0) for k in basic_patterns.keys())
    
    total_patterns = features["total_advanced"] + features["total_medium"] + features["total_basic"]
    features["ratio_advanced"] = features["total_advanced"] / (total_patterns + 1)
    features["ratio_medium"] = features["total_medium"] / (total_patterns + 1)
    features["ratio_basic"] = features["total_basic"] / (total_patterns + 1)
    
    return features

# PREDICTION FUNCTION
def predict_difficulty(title, description, input_desc, output_desc):
    combined_text = (
        title + " " +
        title + " " +
        description + " " +
        input_desc + " " +
        output_desc
    )
    cleaned_text = clean_text(combined_text)
    feature_dict = extract_features(cleaned_text)
    X_manual = np.array([[feature_dict.get(col, 0) for col in feature_cols]])
    X_tfidf = tfidf.transform([cleaned_text])
    X_manual_scaled = scaler.transform(X_manual)
    X_combined = hstack([X_tfidf, X_manual_scaled])
    class_pred = clf.predict(X_combined)[0]
    class_proba = clf.predict_proba(X_combined)[0]
    class_name = le.inverse_transform([class_pred])[0]
    confidence = class_proba[class_pred]
    class_pred = clf.predict(X_combined)[0]
    reg = regressors[int(class_pred)]
    score_pred = reg.predict(X_combined)[0]
    score_pred = float(np.clip(score_pred, 1.0, 5.0))
    insights = []
    
    # Algorithm detection
    if feature_dict.get("total_advanced", 0) > 0:
        insights.append("Advanced algorithms detected (DP, Graph algorithms)")
    if feature_dict.get("total_medium", 0) > 0:
        insights.append("Medium complexity patterns found")
    if feature_dict.get("has_large_n", 0):
        insights.append(f"Large constraint detected (n ≥ 10⁵)")
    if feature_dict.get("has_huge_n", 0):
        insights.append(f"Very large constraint (n ≥ 10⁶)")
    
    # Problem type
    if feature_dict.get("asks_optimal", 0):
        insights.append("Optimization problem")
    if feature_dict.get("asks_count", 0):
        insights.append("Counting problem")
    if feature_dict.get("matrix_input", 0):
        insights.append("Matrix/Grid problem")
    
    # Complexity
    word_count = feature_dict.get("word_count", 0)
    if word_count > 200:
        insights.append("Complex problem statement")
    elif word_count < 50:
        insights.append("Concise problem statement")
    
    if not insights:
        insights.append("Standard programming problem")
    
    return {
        'class': class_name.capitalize(),
        'score': round(float(score_pred), 2),
        'confidence': round(float(confidence * 100), 2),
        'probabilities': {
            le.classes_[i].capitalize(): round(float(class_proba[i] * 100), 2)
            for i in range(len(le.classes_))
        },
        'insights': insights,
        'stats': {
            'word_count': feature_dict.get("word_count", 0),
            'max_constraint': feature_dict.get("max_number", 0),
            'complexity_indicators': feature_dict.get("total_advanced", 0) + feature_dict.get("total_medium", 0)
        }
    }

# ROUTES
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        title = request.form.get("title", "")
        description = request.form.get("description", "")
        input_desc = request.form.get("input_desc", "")
        output_desc = request.form.get("output_desc", "")
        
        if not any([title, description, input_desc, output_desc]):
            return jsonify({
                'error': 'Please provide at least one field (title or description)'
            }), 400
        
        result = predict_difficulty(title, description, input_desc, output_desc)
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'features': len(feature_cols),
        'classes': list(le.classes_)
    })

# ERROR HANDLERS
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
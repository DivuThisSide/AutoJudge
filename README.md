# AutoJudge – Programming Problem Difficulty Predictor

## Project Overview
**AutoJudge** is a Machine Learning–based web application that predicts the **difficulty level** (Easy / Medium / Hard) and an associated **difficulty score (1–5)** for competitive programming problems based solely on their problem statements.

The project aims to automate difficulty estimation using **text analysis, feature engineering, and hybrid ML models**, providing insights similar to online judge platforms.

---

## Dataset Used
- **Source:** Aggregated competitive programming problems  
- **Format:** JSON Lines (`.jsonl`)
- **Size:** ~4,100 programming problems
- **Fields used:**
  - `title`
  - `description`
  - `input_description`
  - `output_description`
  - `problem_class` (easy / medium / hard)
  - `problem_score` (numeric difficulty score)

Each problem includes both **categorical difficulty labels** and **continuous score values**, enabling classification and regression.

---

## Approach & Models Used

### 1️⃣ Text Processing
- Lowercasing, punctuation removal
- Combined title + description + input/output text
- Title weighted more heavily than description

### 2️⃣ Feature Engineering
- **TF-IDF (1–3 grams)** for textual representation
- Hand-crafted numerical features:
  - Text length, word count, sentence count
  - Constraint detection (`n ≥ 10⁵`, `n ≥ 10⁶`)
  - Mathematical symbols and numeric patterns
  - Algorithm indicators (DP, graph, BFS/DFS, binary search, etc.)

### 3️⃣ Classification (Difficulty Prediction)
- **Gradient Boosting Classifier**
- Class imbalance handled using **SMOTE**
- Outputs:
  - Predicted class
  - Class probabilities (confidence)

### 4️⃣ Regression (Score Prediction)
- **Class-conditioned Random Forest Regressors**
- Separate regressor trained for:
  - Easy problems
  - Medium problems
  - Hard problems
- Ensures realistic score prediction and prevents score collapse

---

## 📊 Evaluation Metrics

### Classification
- **Accuracy**
- **Confusion Matrix**
- **Precision, Recall, F1-Score**

### Regression
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

> Class-conditioned regression significantly improves score stability and interpretability compared to a single global regressor.

---

## 🖥️ Web Application (Flask)

The project includes a Flask-based web interface where users can:

- Enter:
  - Problem title
  - Description
  - Input description
  - Output description
- Get:
  - Predicted difficulty (Easy / Medium / Hard)
  - Difficulty score (1–5)
  - Confidence percentage
  - Algorithmic insights (e.g., “Advanced algorithms detected”)

### UI Highlights
- Difficulty badge (color-coded)
- Score progress meter
- Insight explanations
- Responsive design using CSS

---

## 🚀 Steps to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone <your-github-repo-link>
cd AutoJudge

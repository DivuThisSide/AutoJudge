# AutoJudge – Programming Problem Difficulty & Score Predictor

## Project Overview
**AutoJudge** is an end-to-end Machine Learning–based web application that predicts the **difficulty level** (*Easy / Medium / Hard*) and a corresponding **difficulty score (1–5)** for competitive programming problems using only the problem statement text.

The project combines **natural language processing, feature engineering, ensemble classification, and class-conditioned regression** to automatically estimate problem difficulty in a way similar to online judge platforms.

---

## Dataset Used
- **Source:** Competitive programming problems (aggregated)
- **Format:** JSON Lines (`.jsonl`)
- **Total problems:** 4,112
- **Fields used:**
  - `title`
  - `description`
  - `input_description`
  - `output_description`
  - `problem_class` (easy / medium / hard)
  - `problem_score` (numeric difficulty score)

The dataset contains both **categorical difficulty labels** and **continuous score values**, enabling a hybrid ML approach.

---

## 🧠 Approach & Models Used

### 1️⃣ Text Preprocessing
- Lowercasing and normalization
- Removal of special characters
- Combined problem text:
  - Title (weighted ×2)
  - Description
  - Input description
  - Output description

---

### 2️⃣ Feature Engineering
#### 🔹 Text Features
- **TF-IDF (1–3 grams)** with sublinear scaling
- Stop-word removal

#### 🔹 Handcrafted Features
- Text length, word count, sentence statistics
- Numeric constraint detection (`n ≥ 10⁵`, `n ≥ 10⁶`)
- Mathematical symbol density
- Algorithmic pattern detection:
  - Dynamic Programming
  - Graph algorithms
  - BFS / DFS
  - Binary search
- Aggregate complexity indicators

All numeric features are standardized using **StandardScaler**.

---

### 3️⃣ Classification (Difficulty Prediction)
- Models trained:
  - Gradient Boosting Classifier
  - Random Forest Classifier
  - Support Vector Machine (RBF)
- **SMOTE** used to handle class imbalance
- **Ensemble model** built using weighted probability averaging
- Best-performing classifier selected automatically

---

### 4️⃣ Regression (Score Prediction)
- **Class-conditioned Random Forest Regression**
- Separate regressor trained for:
  - Easy problems
  - Medium problems
  - Hard problems
- Prevents score collapse and ensures realistic difficulty scoring
- Final scores are clipped to valid range **[1, 5]**

---

## 📊 Evaluation Metrics

### 🔹 Classification Performance
- **Accuracy:** **54.92%**
- Confusion matrix and per-class precision/recall reported

| Class   | Precision | Recall | F1-score |
|--------|----------|--------|---------|
| Easy   | 0.52 | 0.41 | 0.45 |
| Medium | 0.45 | 0.32 | 0.38 |
| Hard   | 0.60 | 0.77 | 0.67 |

> Medium problems show lower recall due to inherent overlap with Easy and Hard classes.

---

### 🔹 Regression Performance
- **MAE:** **1.366**
- **RMSE:** **1.775**

These results are reasonable given the subjective nature of difficulty scoring and the absence of solution code.

---

## 🖥️ Web Application (Flask)

The project includes a **Flask-based web interface** where users can:

### 🔹 Input
- Problem title
- Problem description
- Input description
- Output description

### 🔹 Output
- Predicted difficulty (Easy / Medium / Hard)
- Difficulty score (1–5)
- Confidence percentage
- Algorithmic insights (e.g., advanced patterns, constraints detected)

### 🔹 UI Features
- Color-coded difficulty badges
- Difficulty progress meter
- Explainable insights section
- Responsive layout using CSS

---

## 🚀 Steps to Run the Project Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/DivuThisSide/AutoJudge.git
cd AutoJudge
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Models
```bash
python train.py
```

### 4️⃣ Run the Flask Web App
```bash
python app.py
```
Open your browser and visit: http://127.0.0.1:5000

---

## Demo Video

Project Demo (2–3 minutes): https://drive.google.com/file/d/1Bufb7IJDuptMSPO5A2aYKaDaQht9ocyw/view?usp=sharing

The demo covers:
- **Project Overview**
- **Model Predictions**
- **Web Interface Usage**
- **Example problem evaluations**

---

## Author
- **Name:** Divyansh Bansal
- **Branch:** Data Science and Artificial Intelligence (DSAI)
- **Department:** Mehta Family School for Data Science and Artificial Intelligence (M.F.S.DSAI)
- **Year** 2nd (2024-28)
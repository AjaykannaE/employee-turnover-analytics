# Employee Turnover Analytics
### Machine Learning Project — Portobello Tech

**Author:** Ajaykanna E | **Course:** Machine Learning using Python

---

## Problem Background

Portobello Tech periodically evaluates employee work details to predict turnover.
The HR Department uses this data to identify at-risk employees and plan targeted retention strategies.

**Employee Turnover** refers to the total number of workers who leave a company over time. Predicting it early allows HR to intervene before talent is lost — reducing recruitment costs, preserving institutional knowledge, and improving overall workforce stability.

---

## Dataset

**File:** `HR_comma_sep.csv` — 14,999 employee records with 10 features

| Column | Description |
|--------|-------------|
| `satisfaction_level` | Employee satisfaction score (0–1) |
| `last_evaluation` | Performance evaluation score (0–1) |
| `number_project` | Number of projects assigned |
| `average_montly_hours` | Average monthly hours at office |
| `time_spend_company` | Years spent in the company |
| `Work_accident` | 0 = No accident, 1 = Had accident |
| `left` | **Target** — 0 = Stayed, 1 = Left |
| `promotion_last_5years` | Promoted in last 5 years? |
| `sales` | Department name (renamed to `department`) |
| `salary` | Salary level (low / medium / high) |

---

## Project Structure

```
├── Employee_Turnover_Analytics_Final_3.ipynb   # Main notebook
├── HR_comma_sep.csv                             # HR dataset
└── README.md
```

---

## Tasks Covered

### Task 1 — Data Quality Check
- Verified zero missing values across all columns
- Detected and removed duplicate rows
- Dataset confirmed clean and ready for analysis

### Task 2 — Exploratory Data Analysis (EDA)

**Correlation Heatmap:**
- `satisfaction_level` has the strongest negative correlation with `left` (-0.39) — the most predictive feature
- `time_spend_company` shows moderate positive correlation (0.17)

**Distribution Analysis — all three key features are bimodal:**

| Feature | Finding |
|---------|---------|
| Satisfaction Level | Two groups: very dissatisfied (0.1–0.2) and satisfied (0.7–0.9) |
| Last Evaluation | Two groups: average performers (0.5–0.6) and high performers (0.8–1.0) |
| Monthly Hours | Two groups: normal workload (~150 hrs) and overloaded (~260 hrs) |

**Project Count vs Turnover:**
- U-shaped relationship — employees with **2 projects** (disengaged) and **6–7 projects** (burned out) both show high turnover
- **3–4 projects** is the optimal workload range with lowest attrition

### Task 3 — K-Means Clustering of Employees Who Left
Applied K-Means (k=3) on satisfaction level and last evaluation for all employees who left:

| Cluster | Satisfaction | Evaluation | Profile | Likely Reason for Leaving |
|---------|-------------|------------|---------|---------------------------|
| **0** | Very Low (~0.11) | High (~0.87) | High performer, unhappy | Overworked, burned out, lack of recognition |
| **1** | Medium (~0.41) | Low (~0.52) | Average performer, dissatisfied | Disengaged, may have left voluntarily or been let go |
| **2** | High (~0.81) | High (~0.91) | Top performer, satisfied | Likely headhunted or found better opportunities |

### Task 4 — Handling Class Imbalance with SMOTE
- Dataset is imbalanced: **~76% stayed / ~24% left**
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** inside a Pipeline to avoid data leakage
- Stratified 80/20 train-test split (random_state=123)
- One-hot encoding applied after splitting to prevent leakage

### Task 5 — Model Training with 5-Fold Stratified Cross-Validation

Three models trained and evaluated using ROC-AUC as the primary metric:

#### Logistic Regression
- Pipeline: `SMOTE → StandardScaler → LogisticRegression`
- Baseline model; interpretable but limited in capturing non-linear interactions

#### Random Forest Classifier
- Pipeline: `SMOTE → RandomForestClassifier`
- Hyperparameter tuning via GridSearchCV
- **Best params:** `n_estimators=200, max_depth=10, min_samples_leaf=3`
- **Best CV ROC-AUC: 0.9827** 

#### Gradient Boosting Classifier
- Pipeline: `SMOTE → GradientBoostingClassifier`
- Hyperparameter tuning via GridSearchCV
- **Best params:** `learning_rate=0.05, n_estimators=100, max_depth=5`
- Strong sequential boosting approach for bias reduction

### Task 6 — Model Evaluation & Best Model Selection

**Why ROC-AUC?**
ROC-AUC is threshold-independent and ideal for imbalanced classification. Plain accuracy would be misleading here since a model predicting "stayed" for everyone would still score ~76%.

**Why Recall over Precision?**
A **False Negative** (missing an employee who will leave) is far more costly than a False Positive (flagging someone who stays). Missing a leaver means no intervention — resulting in unexpected talent loss and high replacement costs. **Recall maximizes the detection of true leavers.**

**Selected Model: Tuned Random Forest** — best balance of ROC-AUC, Recall, and generalization.

### Task 7 — Retention Strategies via Risk Zone Segmentation

Employees scored by predicted turnover probability and segmented into 4 zones:

| Zone | Probability | Action |
|------|-------------|--------|
| Safe Zone | < 20% | Maintain engagement, periodic check-ins |
| Low-Risk Zone | 20–60% | Monitor closely, career growth conversations |
| Medium-Risk Zone | 60–90% | Immediate manager intervention, workload review |
| High-Risk Zone | > 90% | Critical — compensation review, urgent retention action |

---

## Model Performance Summary

| Model | CV ROC-AUC | Notes |
|-------|------------|-------|
| Logistic Regression | ~0.78 | Baseline |
| Random Forest (Tuned) | **0.9827** | Best Model |
| Gradient Boosting (Tuned) | ~0.98 | Strong alternative |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Models | Logistic Regression, Random Forest, Gradient Boosting |
| Tuning | GridSearchCV |
| Evaluation | ROC-AUC, Confusion Matrix, Classification Report |

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```

3. **Place the dataset** (`HR_comma_sep.csv`) in the same directory as the notebook.

4. **Run the notebook**
   ```bash
   jupyter notebook Employee_Turnover_Analytics_Final_3.ipynb
   ```
   Run all cells from top to bottom.

---

## Key Business Insights

- **Satisfaction level is the #1 predictor** of turnover — HR should prioritize regular satisfaction surveys.
- **Workload sweet spot is 3–4 projects** — under and overloading both drive attrition significantly.
- **Three distinct leaver profiles exist** — each requiring a different retention approach (recognition, engagement, or competitive offers).
- The **Random Forest model with SMOTE** can flag at-risk employees before they resign, enabling proactive intervention.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

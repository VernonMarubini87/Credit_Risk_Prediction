# Credit Risk Prediction

Classification project predicting whether a loan applicant is a good or bad credit risk from their demographic and financial profile, built on the German Credit Data dataset and deployed as a live Streamlit app.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-model-orange)
![Status](https://img.shields.io/badge/status-deployed-brightgreen)

## Business Problem

A lender needs to decide, before approving a loan, whether an applicant is likely to be a good or bad credit risk. This project builds and compares several classification models on applicant data — age, sex, job type, housing, savings/checking account status, credit amount, and loan duration — to support that decision, and wraps the best-performing model in an interactive scoring app.

## Project Structure

```
Credit_Risk_Prediction/
├── README.md
├── requirements.txt
├── runtime.txt
├── .streamlit/
│   └── config.toml
├── Credit_Risk_Modeling.ipynb        # EDA → encoding → model training → comparison
├── app.py                             # Streamlit app for live risk scoring
├── extra_xgb_credit_model.pkl         # Trained XGBoost model
├── Sex_encoder.pkl
├── Housing_encoder.pkl
├── Saving accounts_encoder.pkl
└── Checking account_encoder.pkl
```

The notebook runs end-to-end top to bottom; `app.py` loads the model and encoders it produces.

---

## Exploratory Data Analysis

1,000 applicants, 10 fields plus the `Risk` target (good/bad). Two fields had significant missing data — `Saving accounts` (183 missing) and `Checking account` (394 missing) — which were dropped row-wise, leaving **522 applicants** for modeling (291 good risk, 231 bad risk — a 55.7% / 44.3% split).

Explored age, credit amount, and duration distributions, a numeric correlation heatmap, and categorical breakdowns (Sex, Housing, Saving/Checking account, Purpose) against `Risk`.

**Headline finding:** bad-risk applicants carry noticeably higher average credit amounts (3,881 vs. 2,801 in the dataset's currency units) and longer loan durations (25.4 vs. 18.1 months) than good-risk applicants — the two clearest separating signals in the data.

## Feature Engineering & Encoding

- **Final features:** Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration
- `Purpose` was explored in EDA but not carried into the final feature set
- Categorical fields (Sex, Housing, Saving accounts, Checking account) label-encoded; each encoder saved as a `.pkl` for consistent inference in the app
- Target encoded: bad → 0, good → 1
- 80/20 stratified train/test split (417 train / 105 test)

## Model Training & Comparison

Four classifiers tuned via `GridSearchCV` (5-fold cross-validation), compared on held-out test accuracy:

| Model | Test Accuracy |
|---|---|
| Decision Tree | 58.1% |
| Random Forest | 61.9% |
| Extra Trees | 64.8% |
| **XGBoost** | **67.6%** |

**Headline finding:** XGBoost (`colsample_bytree=0.7, learning_rate=0.1, max_depth=3, n_estimators=200, subsample=1`, with `scale_pos_weight` for class imbalance) was selected as the final model, outperforming the tree-based baselines by 3–10 percentage points. With only 522 usable rows and accuracy as the sole metric measured so far, this is a solid baseline rather than a fully validated score — see [Extending This Project](#extending-this-project).

## Deployment — Streamlit App

`app.py` loads the trained XGBoost model and the saved encoders, takes an applicant's details as form input, and returns a predicted risk classification (good/bad) in real time.

---

## Getting Started

### Option A — Jupyter Notebook

```bash
git clone https://github.com/VernonMarubini87/Credit_Risk_Prediction.git
cd Credit_Risk_Prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Credit_Risk_Modeling.ipynb
```

### Option B — Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

> 🔗 *Add your deployed Streamlit Community Cloud link here once live.*

## Dataset

German Credit Data — 1,000 applicants with demographic and financial attributes (age, sex, job, housing, savings/checking account status, credit amount, duration, purpose) and a binary good/bad risk label.

## Extending This Project

- Report more than accuracy — confusion matrix, precision/recall, F1, and ROC-AUC, so the good-vs-bad risk trade-off is visible for a lending use case
- Impute missing `Saving accounts` / `Checking account` values instead of dropping rows — dropping discarded ~48% of the dataset
- Add SHAP explainability so each prediction shows *why* an applicant was flagged as risky
- Revisit whether `Purpose` (loan reason) adds predictive value as a feature
- Add input validation and confidence scores to the Streamlit app

## Author

Vernon Marubini — [LinkedIn](#) · [GitHub](#)
# Credit_Risk_Prediction

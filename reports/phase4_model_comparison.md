# PHASE 4 — CLASSICAL ML BASELINES EVALUATION REPORT
## AI-Powered Fake Job and Recruitment Scam Detection System

**Degree**: Final-Year B.Tech CSE (AI & ML)  
**Phase**: 4 — Classical Baseline Models & Representation Comparison  
**Date Generated**: 2026-10-06 20:37:30  
**Split**: Verified 80/20 Stratified Split ($N_{\text{train}}=14,304, N_{\text{test}}=3,576$)  

---
## 1. Executive Summary & Experimental Findings

In this phase, **7 classical classifiers** were trained and evaluated across **3 distinct feature representations** (`TEXT_TFIDF`, `HYBRID_ALL`, and `ALL_TABULAR`), resulting in **21 successfully executed experimental pipelines**.

- **Best Model by Minority Fraud F1-Score**: **LinearSVC (Balanced)** under **`TEXT_TFIDF`** (Fraud F1 = **0.8810**, Recall = **0.8555**, Precision = **0.9080**)
- **Best Model by PR-AUC (Average Precision)**: **LinearSVC (Balanced)** under **`HYBRID_ALL`** (PR-AUC = **0.9511**, ROC-AUC = **0.9920**)

---
## 2. Complete Experimental Results Table

| Representation | Model | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 | Macro F1 | ROC-AUC | PR-AUC | Train Time (s) | Inference Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TEXT_TFIDF** | LinearSVC (Balanced) | 0.9888 | 0.9080 | 0.8555 | **0.8810** | 0.9375 | 0.9915 | 0.9433 | 1.79s | 0.0068s |
| **HYBRID_ALL** | LinearSVC (Balanced) | 0.9871 | 0.8470 | 0.8960 | **0.8708** | 0.9320 | 0.9920 | 0.9511 | 44.53s | 0.0062s |
| **HYBRID_ALL** | RandomForest (Balanced) | 0.9866 | 0.9630 | 0.7514 | **0.8442** | 0.9186 | 0.9904 | 0.9214 | 3.29s | 0.2450s |
| **ALL_TABULAR** | RandomForest (Balanced) | 0.9852 | 0.9545 | 0.7283 | **0.8262** | 0.9092 | 0.9915 | 0.9153 | 1.31s | 0.0724s |
| **TEXT_TFIDF** | RandomForest (Balanced) | 0.9815 | 0.9421 | 0.6590 | **0.7755** | 0.8829 | 0.9889 | 0.8923 | 3.47s | 0.2324s |
| **TEXT_TFIDF** | LogisticRegression (Balanced) | 0.9718 | 0.6513 | 0.8960 | **0.7543** | 0.8696 | 0.9912 | 0.9096 | 0.20s | 0.0050s |
| **HYBRID_ALL** | XGBoost (Weighted) | 0.9723 | 0.6652 | 0.8613 | **0.7506** | 0.8680 | 0.9866 | 0.8954 | 65.37s | 0.0286s |
| **HYBRID_ALL** | LightGBM (Weighted) | 0.9684 | 0.6181 | 0.9075 | **0.7354** | 0.8593 | 0.9889 | 0.9009 | 12.54s | 0.0199s |
| **HYBRID_ALL** | LogisticRegression (Balanced) | 0.9662 | 0.5970 | 0.9249 | **0.7256** | 0.8538 | 0.9900 | 0.8993 | 16.45s | 0.0064s |
| **TEXT_TFIDF** | LightGBM (Weighted) | 0.9642 | 0.5862 | 0.8844 | **0.7051** | 0.8430 | 0.9845 | 0.8775 | 13.52s | 0.0204s |
| **ALL_TABULAR** | LightGBM (Weighted) | 0.9603 | 0.5552 | 0.9017 | **0.6872** | 0.8330 | 0.9841 | 0.8695 | 0.21s | 0.0162s |
| **TEXT_TFIDF** | XGBoost (Weighted) | 0.9603 | 0.5564 | 0.8844 | **0.6830** | 0.8309 | 0.9827 | 0.8671 | 63.88s | 0.0333s |
| **ALL_TABULAR** | XGBoost (Weighted) | 0.9522 | 0.5032 | 0.9075 | **0.6474** | 0.8109 | 0.9851 | 0.8738 | 0.60s | 0.0139s |
| **TEXT_TFIDF** | MultinomialNB | 0.9695 | 0.8721 | 0.4335 | **0.5792** | 0.7817 | 0.9496 | 0.6701 | 0.02s | 0.0150s |
| **ALL_TABULAR** | LinearSVC (Balanced) | 0.8996 | 0.3094 | 0.8728 | **0.4569** | 0.7008 | 0.9463 | 0.5738 | 9.87s | 0.0011s |
| **ALL_TABULAR** | LogisticRegression (Balanced) | 0.8982 | 0.3071 | 0.8786 | **0.4551** | 0.6995 | 0.9566 | 0.6002 | 2.01s | 0.0014s |
| **ALL_TABULAR** | MultinomialNB | 0.9074 | 0.2435 | 0.4335 | **0.3119** | 0.6311 | 0.8020 | 0.2288 | 0.01s | 0.0035s |
| **HYBRID_ALL** | MultinomialNB | 0.9245 | 0.2744 | 0.3410 | **0.3041** | 0.6321 | 0.8694 | 0.2805 | 0.02s | 0.0190s |
| **TEXT_TFIDF** | DummyClassifier (Majority) | 0.9516 | 0.0000 | 0.0000 | **0.0000** | 0.4876 | 0.5000 | 0.0484 | 0.00s | 0.0004s |
| **HYBRID_ALL** | DummyClassifier (Majority) | 0.9516 | 0.0000 | 0.0000 | **0.0000** | 0.4876 | 0.5000 | 0.0484 | 0.00s | 0.0006s |
| **ALL_TABULAR** | DummyClassifier (Majority) | 0.9516 | 0.0000 | 0.0000 | **0.0000** | 0.4876 | 0.5000 | 0.0484 | 0.00s | 0.0006s |

---
## 3. Representation Ablation Comparison (TF-IDF vs HYBRID vs TABULAR)

A critical scientific question of Phase 4 is evaluating whether multimodal fusion of tabular, domain missingness, and heuristic features improves upon text-only TF-IDF representations.

### Key Observations:
1. **Impact of Hybrid Metadata**: Combining tabular and heuristic indicators with TF-IDF significantly reduces false positives and improves precision across gradient boosting trees (XGBoost, LightGBM) and Random Forests.
2. **Linear Models on High-Dimensional Text**: Linear models (`LogisticRegression` and `LinearSVC`) perform strongly on pure `TEXT_TFIDF` due to high linear separability in 10,000-dimensional sparse space.
3. **Tabular-Only Performance**: `ALL_TABULAR` (378 features) demonstrates that structural metadata alone captures substantial fraud signal, but text content is essential to achieve >0.85 F1 scores.

---
## 4. Diagnostic Visualizations Generated

- `reports/figures/confusion_matrix_leading_model.png`: Detailed error analysis breakdown (TP, FP, TN, FN).
- `reports/figures/model_metrics_comparison.png`: Comprehensive bar chart across leading classifiers.
- `reports/figures/roc_curves_comparison.png`: ROC comparison across models.
- `reports/figures/pr_curves_comparison.png`: Precision-Recall curves evaluating minority class ranking.
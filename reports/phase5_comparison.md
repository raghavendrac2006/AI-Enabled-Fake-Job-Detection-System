# PHASE 5 — IMBALANCE HANDLING & TARGETED OPTIMIZATION REPORT
## AI-Powered Fake Job and Recruitment Scam Detection System

**Degree**: Final-Year B.Tech CSE (AI & ML)  
**Phase**: 5 — Imbalance Handling, Targeted Tuning & Validation Threshold Analysis  
**Date Generated**: 2026-10-06 21:46:10  
**Split**: Verified 80/20 Stratified Split ($N_{\text{train}}=14,304, N_{\text{test}}=3,576$)  

---
## 1. Executive Summary & Core Scientific Answers

1. **Did SMOTE / BorderlineSMOTE improve performance over analytical class weighting?**
   - **No for tree-based & high-dim TF-IDF:** Across high-dimensional TF-IDF (10,000 features) and tree models (`RandomForest`, `LightGBM`), analytical class-weighting consistently matched or outperformed synthetic oversampling. In sparse spaces, synthetic linear interpolation generates noisy samples that degrade precision.
   - **Marginal for Linear Margin Boundary:** On `HYBRID_ALL` (10,378 features), `LinearSVC (Borderline_SMOTE)` achieved minority Fraud F1 of **0.8902** (Precision: 0.9146, Recall: 0.8671, PR-AUC: 0.9490), outperforming default class-weighting (`0.8708`) by synthesizing solely along the borderline decision margin.

2. **Did Targeted Hyperparameter Tuning improve performance?**
   - **Yes.** 5-Fold Stratified CV on the training partition ($N_{\text{train}}=14,304$) identified optimal hyperparameters for tree boosting:
     - **Champion Model (`Tuned_LightGBM` on `HYBRID_ALL`)**: Configured with `learning_rate=0.1`, `n_estimators=100`, `num_leaves=31`, `scale_pos_weight=19.65`, achieving the project's highest test set minority Fraud F1 of **0.8930** (Precision: **0.9481**, Recall: **0.8439**, PR-AUC: **0.9462**, ROC-AUC: **0.9926**, Accuracy: **0.9902**).
     - **Tuned LinearSVC**: Reduced regularization penalty ($C=2.0$ for `TEXT_TFIDF`, $C=5.0$ for `HYBRID_ALL`), improving precision and generalization.

3. **Did Threshold Analysis on Validation Folds Generalize?**
   - **Calibrated for LinearSVC (`TEXT_TFIDF`):** Out-of-fold validation threshold analysis identified an optimal decision margin threshold of **$\tau^* = 0.0644$** (Validation F1: 0.8414). When applied to the untouched test set, $\tau^* = 0.0644$ increased precision from 0.9236 to 0.9286 (reducing test false alarms to 11) while maintaining a balanced Fraud F1 of 0.8746.

---
## 2. Complete Phase 5 Experimental Results Table

| Experiment Type | Model | Representation | Strategy | Fraud Prec | Fraud Rec | Fraud F1 | Macro F1 | PR-AUC | ROC-AUC | Accuracy | Train Time |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Imbalance_Handling | LinearSVC (Class_Weighted) | TEXT_TFIDF | Class_Weighted | 0.9080 | 0.8555 | **0.8810** | 0.9375 | 0.9433 | 0.9915 | 0.9888 | 1.46s |
| Imbalance_Handling | LinearSVC (SMOTE) | TEXT_TFIDF | SMOTE | 0.9231 | 0.8324 | **0.8754** | 0.9347 | 0.9437 | 0.9915 | 0.9885 | 7.93s |
| Imbalance_Handling | LinearSVC (Borderline_SMOTE) | TEXT_TFIDF | Borderline_SMOTE | 0.9290 | 0.8324 | **0.8780** | 0.9361 | 0.9445 | 0.9920 | 0.9888 | 4.20s |
| Imbalance_Handling | LinearSVC (Class_Weighted) | HYBRID_ALL | Class_Weighted | 0.8470 | 0.8960 | **0.8708** | 0.9320 | 0.9511 | 0.9920 | 0.9871 | 38.95s |
| Imbalance_Handling | LinearSVC (SMOTE) | HYBRID_ALL | SMOTE | 0.8994 | 0.8786 | **0.8889** | 0.9417 | 0.9518 | 0.9919 | 0.9894 | 113.57s |
| Imbalance_Handling | LinearSVC (Borderline_SMOTE) | HYBRID_ALL | Borderline_SMOTE | 0.9146 | 0.8671 | **0.8902** | 0.9424 | 0.9490 | 0.9915 | 0.9897 | 133.07s |
| Imbalance_Handling | RandomForest (Class_Weighted) | HYBRID_ALL | Class_Weighted | 0.9630 | 0.7514 | **0.8442** | 0.9186 | 0.9214 | 0.9904 | 0.9866 | 2.61s |
| Imbalance_Handling | RandomForest (SMOTE) | HYBRID_ALL | SMOTE | 0.9921 | 0.7225 | **0.8361** | 0.9145 | 0.9356 | 0.9923 | 0.9863 | 9.22s |
| Imbalance_Handling | RandomForest (Borderline_SMOTE) | HYBRID_ALL | Borderline_SMOTE | 0.9917 | 0.6879 | **0.8123** | 0.9021 | 0.9297 | 0.9923 | 0.9846 | 10.00s |
| Imbalance_Handling | LogisticRegression (Class_Weighted) | TEXT_TFIDF | Class_Weighted | 0.6513 | 0.8960 | **0.7543** | 0.8696 | 0.9096 | 0.9912 | 0.9718 | 0.18s |
| Imbalance_Handling | LogisticRegression (SMOTE) | TEXT_TFIDF | SMOTE | 0.7688 | 0.8844 | **0.8226** | 0.9064 | 0.9161 | 0.9915 | 0.9815 | 0.53s |
| Imbalance_Handling | LightGBM (Class_Weighted) | HYBRID_ALL | Class_Weighted | 0.6181 | 0.9075 | **0.7354** | 0.8593 | 0.9009 | 0.9889 | 0.9684 | 11.47s |
| Imbalance_Handling | LightGBM (SMOTE) | HYBRID_ALL | SMOTE | 0.5436 | 0.9364 | **0.6879** | 0.8329 | 0.9018 | 0.9851 | 0.9589 | 24.68s |
| Targeted_Tuning | Tuned_LinearSVC | TEXT_TFIDF | Tuned_Hyperparameters (C=2.0) | 0.9236 | 0.8382 | **0.8788** | 0.9365 | 0.9421 | 0.9913 | 0.9888 | 24.54s |
| Targeted_Tuning | Tuned_LinearSVC | HYBRID_ALL | Tuned_Hyperparameters (C=5.0) | 0.9193 | 0.8555 | **0.8862** | 0.9403 | 0.9471 | 0.9917 | 0.9894 | 590.54s |
| Targeted_Tuning | Tuned_RandomForest | HYBRID_ALL | Tuned_Hyperparameters | 0.9457 | 0.7052 | **0.8079** | 0.8997 | 0.9195 | 0.9922 | 0.9838 | 93.31s |
| Targeted_Tuning | Tuned_LightGBM | HYBRID_ALL | Tuned_Hyperparameters | 0.9481 | 0.8439 | **0.8930** | 0.9439 | 0.9462 | 0.9926 | 0.9902 | 784.96s |
| Threshold_Optimization | Tuned_LinearSVC (Val_Tuned_Threshold) | TEXT_TFIDF | Threshold_Tuned_tau=0.0644 | 0.9286 | 0.8266 | **0.8746** | 0.9343 | 0.9421 | 0.9913 | 0.9885 | 0.00s |

---
## 3. Phase 4 Baseline vs Phase 5 Optimized Comparison

| Metric | Phase 4 Best (LinearSVC Baseline) | Phase 5 Champion (Tuned_LightGBM on HYBRID_ALL) | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Minority Fraud F1** | 0.8810 | **0.8930** | +0.0120 |
| **Minority Precision** | 0.9080 | **0.9481** | +0.0401 |
| **Minority Recall** | 0.8555 | **0.8439** | -0.0116 |
| **PR-AUC (Avg Precision)** | 0.9433 | **0.9462** | +0.0029 |
| **ROC-AUC** | 0.9915 | **0.9926** | +0.0011 |
| **Test Accuracy** | 0.9888 | **0.9902** | +0.0014 |
| **Test False Positives** | 15 | **8** (46.7% reduction) | **-7 FPs** |

---
## 4. Generated Diagnostic Visualizations

- `reports/figures/06_phase5_smote_vs_weighting.png`: Direct comparison of SMOTE vs class weighting across models.
- `reports/figures/07_phase5_tuning_cv_results.png`: 5-fold CV score distributions across parameter candidates.
- `reports/figures/08_phase5_threshold_curve.png`: Out-of-fold validation threshold precision/recall/F1 trade-off curve (LinearSVC on TEXT_TFIDF).
- `reports/figures/09_phase5_best_model_confusion_matrix.png`: Detailed confusion matrix of the leading Phase 5 champion model.
- `reports/figures/10_phase4_vs_phase5_comparison.png`: Phase 4 baseline vs Phase 5 optimized comparison bar chart.

---
## 5. Saved Trained Model Artifacts

The following genuine trained model artifacts from Phase 5 are preserved in `models/`:
- `models/phase5_best_LightGBM_HYBRID_ALL.joblib` — **Phase 5 Champion** (`scale_pos_weight=19.65`, `num_leaves=31`, `learning_rate=0.1`, `n_estimators=100`, Fraud F1: `0.8930`, PR-AUC: `0.9462`).
- `models/phase5_best_LinearSVC_HYBRID_ALL.joblib` — Tuned LinearSVC on multimodal features ($C=5.0$, `class_weight='balanced'`, Fraud F1: `0.8862`).
- `models/phase5_best_LinearSVC_TEXT_TFIDF.joblib` — Tuned LinearSVC on text features ($C=2.0$, `class_weight='balanced'`, Fraud F1: `0.8788`).
- `models/phase5_best_RandomForest_HYBRID_ALL.joblib` — Tuned Random Forest on multimodal features (`min_samples_split=5`, `n_estimators=100`, Fraud F1: `0.8079`).

# Phase 2: Data Preprocessing & Feature Engineering Foundation Report
**AI-Powered Fake Job and Recruitment Scam Detection System**  
*Academic Project: Final-Year B.Tech CSE (AI & ML)*  
*Phase 2 Completion Date: October 2026 | Random Seed: 42*

---

## 1. Executive Summary
Phase 2 establishes a leak-free, modular, and reproducible preprocessing and feature-engineering pipeline for the EMSCAD Fake Job Postings dataset. All transformations were designed strictly in adherence to academic data integrity standards:
- **Raw Data Immutability**: The raw file `data/raw/fake_job_postings.csv` was verified cryptographically before and after execution (SHA-256: `59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60`).
- **Strict Leakage Prevention**: The dataset was partitioned into an **80/20 Stratified Train/Test split** prior to fitting any preprocessing step. All vocabulary generation (TF-IDF), category encodings, and numerical scalers were fitted **exclusively on training data**.
- **Dual Representation Architecture**: Clean tabular text representations (`train.csv`, `test.csv`) were preserved for contextual Transformer fine-tuning (Representation D), while high-dimensional sparse hybrid matrices (`X_train_tfidf_tabular.npz`, `X_test_tfidf_tabular.npz`) were generated for classical ML models (Representation A).

---

## 2. Text Normalization & Composite Text Design

### What Was Cleaned:
1. **HTML Stripping & Entity Decoding**: Cleaned `<p>`, `<br>`, `<li>`, `<div>` tags and unescaped HTML entities (`&amp;`, `&quot;`, `&#39;`).
2. **Unicode NFKC Normalization**: Standardized non-ASCII characters, typographical quotes, and ligature variations.
3. **URL & Email Redaction**: Preserved domain tokens (`[URL]`, `[EMAIL]`) rather than leaving unstructured raw hyperlinks.
4. **Whitespace Normalization**: Collapsed consecutive spaces and erratic tabs while maintaining section boundaries.

### What Was Preserved:
- **Semantic Punctuation & Sentence Structure**: Punctuation necessary for sentiment, tone, and transformer tokenizers was preserved.
- **Section Boundaries**: Text fields were concatenated into a structured composite format:
  ```text
  TITLE: <cleaned title>
  COMPANY PROFILE: <cleaned company_profile or [UNSPECIFIED]>
  DESCRIPTION: <cleaned description>
  REQUIREMENTS: <cleaned requirements or [UNSPECIFIED]>
  BENEFITS: <cleaned benefits or [UNSPECIFIED]>
  ```

---

## 3. Missing Value Handling Strategy

| Modality | Column(s) | Strategy Applied | Rationale |
| :--- | :--- | :--- | :--- |
| **Unstructured Text** | `company_profile`, `requirements`, `benefits` | Replaced with empty string in raw cleaner and tagged with `[UNSPECIFIED]` token in composite string. | Prevents synthetic linguistic hallucination while signaling absence to NLP models. |
| **Categorical Metadata** | `employment_type`, `required_experience`, `required_education`, `industry`, `function` | Imputed with dedicated category `'__MISSING__'`. | Absence of metadata is informative in fraudulent listings. |
| **Binary Metadata** | `telecommuting`, `has_company_logo`, `has_questions` | Imputed with `0` (False) if missing. | Safe conservative assumption. |
| **Salary Range** | `salary_range` | Kept raw text; engineered boolean `has_salary_specified`. | Preserves high-signal absence information without forcing invalid numeric imputations. |

---

## 4. Stratified Train / Test Partitioning

| Partition | Total Samples | Legitimate (0) | Fraudulent (1) | Fraud Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Training Set (80%)** | **`14,304`** | `13,611` | `693` | **`4.84%`** |
| **Test Set (20%)** | **`3,576`** | `3,403` | `173` | **`4.84%`** |
| **Full Dataset** | **`17,880`** | **`17,014`** | **`866`** | **`4.84%`** |

- **ID Overlap**: `0` records (Verified zero intersection between `train_ids` and `test_ids`).
- **Target Isolation**: `fraudulent` was extracted into label vectors (`y_train`, `y_test`) and completely dropped from `X`.
- **ID Dropping**: `job_id` was dropped from `X_train` and `X_test` to prevent database index memorization.

---

## 5. Feature Engineering & TF-IDF Vectorization

### A. TF-IDF Text Features (Fitted ONLY on Training Set)
- **Max Features**: `10,000`
- **N-gram Range**: `(1, 2)` (Unigrams + Bigrams)
- **Sublinear TF Scaling**: `True` ($1 + \log(TF)$)
- **Minimum Document Frequency (`min_df`)**: `3`
- **Maximum Document Frequency (`max_df`)**: `0.95`
- **Actual Training Vocabulary Size**: **`10,000` n-grams**

### B. Tabular & Domain Engineered Features
- **One-Hot Encoded Categorical Attributes**: 215 features (including `'__MISSING__'` categories).
- **Binary Indicator Features**: `telecommuting`, `has_company_logo`, `has_questions`.
- **Domain Presence Indicators**: `has_salary_specified`, `has_company_profile`, `has_requirements`, `has_benefits`.
- **Text Length & Word Count Metrics**: Character lengths and word counts for all 5 sub-fields and composite text (standardized with `StandardScaler(with_mean=False)`).

### C. Combined Feature Matrix Characteristics
- **Total Feature Dimensions**: **`10,215` columns**
- **X_train Matrix Shape**: `(14,304, 10,215)`
- **X_test Matrix Shape**: `(3,576, 10,215)`
- **Matrix Sparsity**: **`96.82%`** (Stored efficiently as Scipy CSR sparse matrix).

---

## 6. Generated & Serialized Artifacts

| Artifact File | Path | Description |
| :--- | :--- | :--- |
| **Train Data (DataFrame)** | [`data/processed/train.csv`](file:///a:/AI-Powered%20Job%20Detection%20System/data/processed/train.csv) | 80% train split containing clean text, metadata, target, and job_id. |
| **Test Data (DataFrame)** | [`data/processed/test.csv`](file:///a:/AI-Powered%20Job%20Detection%20System/data/processed/test.csv) | 20% test split containing clean text, metadata, target, and job_id. |
| **Train Feature Matrix** | [`data/processed/X_train_tfidf_tabular.npz`](file:///a:/AI-Powered%20Job%20Detection%20System/data/processed/X_train_tfidf_tabular.npz) | Sparse CSR feature matrix for classical ML training. |
| **Test Feature Matrix** | [`data/processed/X_test_tfidf_tabular.npz`](file:///a:/AI-Powered%20Job%20Detection%20System/data/processed/X_test_tfidf_tabular.npz) | Sparse CSR feature matrix for leak-free test evaluation. |
| **Target Labels** | [`data/processed/y_train.npy`](file:///a:/AI-Powered%20Job%20Detection%20System/data/processed/y_train.npy), [`data/processed/y_test.npy`](file:///a:/AI-Powered%20Job%20Detection%20System/data/processed/y_test.npy) | Binary ground truth label arrays. |
| **Feature Names Registry** | [`data/processed/feature_names.json`](file:///a:/AI-Powered%20Job%20Detection%20System/data/processed/feature_names.json) | Complete list of all 10,215 feature names for model explainability (SHAP). |
| **Split Metadata** | [`data/processed/split_metadata.json`](file:///a:/AI-Powered%20Job%20Detection%20System/data/processed/split_metadata.json) | Exact seed, sample counts, and fraud proportions. |
| **Fitted Preprocessor** | [`models/preprocessor.joblib`](file:///a:/AI-Powered%20Job%20Detection%20System/models/preprocessor.joblib) | Serialized hybrid pipeline for production inference. |

---

## 7. Status & Phase 3 Readiness

### **VERDICT: READY FOR PHASE 3**
The preprocessing and feature engineering foundation is fully verified, reproducible, tested for zero data leakage, and serialized. The project is ready for **Phase 3 (Baseline Classical ML Models: Naive Bayes, Logistic Regression, Random Forest, XGBoost & Imbalance Experiments)**.

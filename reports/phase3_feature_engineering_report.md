# PHASE 3 — FEATURE ENGINEERING & MULTI-REPRESENTATION FOUNDATION REPORT
## AI-Powered Fake Job and Recruitment Scam Detection System

**Academic Degree**: Final-Year B.Tech in Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
**Phase**: 3 — Feature Engineering & Representation Framework  
**Dataset**: Employment Scam Aegean Dataset (EMSCAD / `fake_job_postings.csv`)  
**Status**: COMPLETED & VERIFIED  

---

## 1. Feature Engineering Objectives

The primary goal of Phase 3 is to construct a modular, leakage-safe, high-dimensional feature engineering architecture that transforms verified tabular and textual job posting records into diverse numerical representations suitable for downstream machine learning and deep learning models.

Key architectural requirements satisfied:
1. **Zero Data Leakage**: All learned transformations (TF-IDF vocabulary, Inverse Document Frequency, One-Hot Encoders, Standard Scalers) are strictly fitted on the 80% training partition ($N=14,304$) and applied as read-only transformations to the test set ($N=3,576$).
2. **Modular Grouping**: Features are partitioned into 8 distinct functional groups, enabling rigorous ablation studies in Phase 4.
3. **Representation Hierarchy**: Concrete implementation of **Representation A (TF-IDF + Tabular Metadata)** with formal abstract contracts established for **Representation B (Word Embeddings)**, **Representation C (Transformer Embeddings)**, and **Representation D (Tokenized Sequences)**.
4. **Heuristic & Domain Integration**: Incorporation of expert recruitment scam patterns (urgent calls to action, fraudulent money transfer terms, free webmail domains, omitted corporate profiles).

---

## 2. Feature Groups Architecture

The system organizes engineered features into 8 primary groups, which can be extracted independently or unified into composite representations:

| Feature Group | Component Class | Num Features | Format | Sparsity | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`TEXT_TFIDF`** | `TfidfVectorizer` | 10,000 | Sparse (`csr_matrix`) | 96.95% | **IMPLEMENTED** |
| **`TEXT_STATISTICS`** | `TextStatisticsExtractor` | 44 | Dense (`ndarray`) | 31.60% | **IMPLEMENTED** |
| **`METADATA_BINARY`** | `MetadataBinaryExtractor` | 3 | Dense (`ndarray`) | 55.75% | **IMPLEMENTED** |
| **`METADATA_CATEGORICAL`** | `MetadataCategoricalExtractor` | 285 | Sparse (`csr_matrix`) | 97.89% | **IMPLEMENTED** |
| **`MISSINGNESS`** | `MetadataBinaryExtractor` | 5 | Dense (`ndarray`) | 28.21% | **IMPLEMENTED** |
| **`SALARY`** | `SalaryFeatureExtractor` | 6 | Dense (`ndarray`) | 89.80% | **IMPLEMENTED** |
| **`CONTACT`** | `ContactFeatureExtractor` | 8 | Dense (`ndarray`) | 99.94% | **IMPLEMENTED** |
| **`HEURISTIC`** | `HeuristicKeywordExtractor` | 27 | Dense (`ndarray`) | 98.47% | **IMPLEMENTED** |
| **`ALL_TABULAR`** | `ModularFeatureBuilder` | 378 | Sparse (`csr_matrix`) | 88.88% | **IMPLEMENTED** |
| **`HYBRID_ALL`** | `ModularFeatureBuilder` | **10,378** | Sparse (`csr_matrix`) | **96.65%** | **IMPLEMENTED** |

---

## 3. Comprehensive Feature Definitions

### 3.1 Text Features (`TEXT_TFIDF` & `TEXT_STATISTICS`)
- **`TEXT_TFIDF` (10,000 features)**:
  - Extracted from `clean_composite_text` with field boundary markers (`TITLE: ...\nCOMPANY PROFILE: ...\nDESCRIPTION: ...\nREQUIREMENTS: ...\nBENEFITS: ...`).
  - Unigram and bigram n-grams ($n \in [1, 2]$), minimum document frequency $\text{min\_df}=3$, maximum document frequency $\text{max\_df}=0.95$, sublinear term frequency scaling ($\text{sublinear\_tf}=\text{True}$), and $L_2$ vector normalization.
- **`TEXT_STATISTICS` (44 features)**:
  - **Per-field Metrics** (for `title`, `company_profile`, `description`, `requirements`, `benefits`):
    - `char_count`: Exact string length.
    - `word_count`: Whitespace token count.
    - `avg_word_length`: `char_count / max(word_count, 1)`.
    - `exclamation_count` & `question_count`: Frequency of sensationalist punctuation.
    - `is_present`: Binary flag for field population.
  - **Composite Metrics**:
    - Total character and word counts across the full ad.
    - `composite_uppercase_count` and `composite_uppercase_ratio` (detects scam 'ALL-CAPS' headlines).
    - `composite_digit_count` and `composite_digit_ratio` (quantifies numeric payouts and telephone digits).
    - `composite_punctuation_count`.
  - **Structural Ratios**:
    - `populated_sections_count`: Sum of non-empty text sections ($[0, 5]$).
    - `desc_to_req_length_ratio` & `desc_to_req_word_ratio`: Laplace-smoothed ratio $\frac{\text{len}(\text{desc}) + 1}{\text{len}(\text{req}) + 1}$.
    - `company_profile_to_desc_ratio`: $\frac{\text{len}(\text{profile}) + 1}{\text{len}(\text{desc}) + 1}$.

### 3.2 Metadata Binary & Categorical (`METADATA_BINARY` & `METADATA_CATEGORICAL`)
- **Binary Attributes (3 features)**:
  - `telecommuting`: Remote work indicator ($0/1$).
  - `has_company_logo`: Company branding indicator ($0/1$).
  - `has_questions`: Presence of screening questions ($0/1$).
- **Categorical One-Hot Encoding (285 features)**:
  - Columns: `employment_type` (8), `required_experience` (8), `required_education` (14), `industry` (132), `function` (38), `country` (85).
  - Explicit missing token `__MISSING__` used to preserve signal in unassigned categorical columns.
  - `OneHotEncoder(handle_unknown="ignore")` ensures zero errors when unseen categories appear during testing or production inference.

### 3.3 Domain Missingness Indicators (`MISSINGNESS`)
- 5 explicit binary indicators capturing systematic omission patterns:
  - `has_company_profile`, `has_requirements`, `has_benefits`, `has_department`, `has_location_specified`.

### 3.4 Salary Features (`SALARY`)
- 6 features safely parsed from `salary_range` with fallback handling for unformatted or irregular compensation strings:
  - `has_salary_specified`: 1 if salary string exists, 0 otherwise.
  - `salary_min` & `salary_max`: Parsed numeric boundaries.
  - `salary_spread`: `salary_max - salary_min`.
  - `is_hourly_scale`: Indicator for low numeric rates ($\le 250$), identifying hourly/daily wage expressions.
  - `is_unrealistic_salary`: Flag for extreme compensation anomalies ($> \$10,000,000$).

### 3.5 Contact & Off-Platform Communication Features (`CONTACT`)
- 8 features detecting attempts to divert candidates to insecure or unmonitored communication channels:
  - `email_count` & `has_email`: General email address detection.
  - `free_email_count` & `has_free_email`: Specific detection of consumer webmail services (`@gmail`, `@yahoo`, `@hotmail`, `@outlook`, `@aol`, `@protonmail`, `@ymail`, `@mail`, `@gmx`).
  - `url_count` & `has_url`: Hyperlink and external URL detection.
  - `phone_count` & `has_phone`: Telephone number detection.

### 3.6 Suspicious Language Heuristics (`HEURISTIC`)
- 27 features derived from externally configured YAML keyword rules (`config/heuristic_keywords.yaml`):
  - **8 Categories**:
    1. `payment_requests`: Wire transfer, Western Union, MoneyGram, cashier check, processing fee.
    2. `urgency`: Urgent, immediate start, act now, apply immediately, instant hire.
    3. `guaranteed_income`: Guaranteed income, earn fast, unlimited earning, financial freedom.
    4. `work_from_home_claims`: Work from anywhere, be your own boss, no experience required.
    5. `personal_info_requests`: Bank details, SSN, credit card, routing number, passport copy.
    6. `external_comms`: Telegram, WhatsApp, Skype interview, Google Hangouts, Wire app.
    7. `cryptocurrency`: Bitcoin, BTC, Ethereum, USDT, wallet address, crypto payout.
    8. `recruitment_fees`: Training fee, application fee, visa fee, equipment purchase.
  - **Metrics per Category**: Count, binary presence flag, and word-normalized frequency ($\frac{\text{count}}{\text{words} + 1}$).
  - **Global Heuristic Aggregates**:
    - `total_scam_keyword_count`: Cumulative hits across all 8 categories.
    - `has_any_scam_keyword`: Global binary flag ($0/1$).
    - `scam_category_diversity`: Number of distinct scam categories triggered in a single posting ($[0, 8]$).

---

## 4. Multi-Representation Framework Roadmap

| Representation Family | Composition | Primary Target Models | Implementation Status |
| :--- | :--- | :--- | :--- |
| **Representation A** | TF-IDF (10,000) + Scaled Tabular (378) = **10,378 features** | Logistic Regression, Naive Bayes, Linear SVM, Random Forest, XGBoost, LightGBM | **IMPLEMENTED** |
| **Representation B** | Dense Word Embeddings (Word2Vec / FastText / GloVe, 300d) + Scaled Tabular (378) | BiLSTM, GRU, CNN-1D with tabular attention | **PLANNED (Phase 5)** |
| **Representation C** | Pretrained Transformer Embeddings (DeBERTa-v3 / RoBERTa pooled 768d) + Tabular | Multimodal MLP, GBDT ensemble on embeddings | **PLANNED (Phase 5)** |
| **Representation D** | Tokenized Text Sequences (Input IDs + Attention Masks, 512 tokens) + Tabular Tensors | End-to-End Fine-Tuned Transformer Backbones | **PLANNED (Phase 5/6)** |

---

## 5. Leakage Prevention and Verification

All feature transformations were subjected to rigorous leak-prevention verification:
1. **Strict Train/Test Isolation**:
   - Training split ($N=14,304$) and test split ($N=3,576$) verified completely disjoint with zero shared `job_id` values ($\text{train\_ids} \cap \text{test\_ids} = \emptyset$).
2. **Identifier & Target Exclusion**:
   - `fraudulent` (target label) and `job_id` (identifier) are strictly excluded from all feature matrices $X$.
3. **Fitted Transformers Isolation**:
   - TF-IDF vocabulary, OneHotEncoder categories, and StandardScaler means/variances were fitted **only on the training split**. Test transformations use already-fitted parameters.
4. **Raw Data Immutability**:
   - Cryptographic SHA-256 hash of `data/raw/fake_job_postings.csv` verified before and after execution:
     `59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60`.

---

## 6. Limitations & Open Experimental Questions

### Current Limitations:
1. **Static Keyword Vulnerability**: Heuristic regex rules capture known scam patterns but can be circumvented by adversarial typosquatting or novel phrasing (to be addressed by Dense Embeddings and Transformer representations in Phase 5).
2. **Salary Sparsity**: Approximately 83% of job postings omit salary information. While `has_salary_specified` provides strong discriminatory power, numerical salary bounds have limited coverage.
3. **High-Cardinality Categoricals**: Location categories (city/state) produce high sparsity if fully one-hot encoded; country-level extraction was adopted to maintain optimal feature density.

### Decisions Requiring Experimental Validation in Phase 4:
- **Ablation Study 1**: Compare TF-IDF alone vs TF-IDF + Tabular features to quantify the empirical boost from metadata.
- **Ablation Study 2**: Evaluate the marginal benefit of Heuristic features vs pure TF-IDF.
- **Ablation Study 3**: Compare tree-based models (XGBoost/LightGBM) performance with dense tabular features vs high-dimensional sparse hybrid representations.

---

## 7. Status Classification

- **IMPLEMENTED**:
  - `src/features/interfaces.py` (Abstract contracts for Representations A, B, C, D)
  - `src/features/text_features.py` (Text statistics and structural ratios)
  - `src/features/salary_features.py` (Safe salary extraction)
  - `src/features/contact_features.py` (Contact and free domain extraction)
  - `src/features/heuristic_features.py` (Scam keyword heuristics)
  - `src/features/metadata_features.py` (Binary, missingness, and one-hot categorical extractors)
  - `src/features/builder.py` (`ModularFeatureBuilder` supporting all 8 groups and hybrid matrices)
  - `src/features/feature_catalog.py` (Feature registry and catalog generator)
  - `config/heuristic_keywords.yaml` (Curated scam keyword dictionaries)
  - `reports/feature_dimensions.json` (Exact dimension and sparsity records)
  - `reports/feature_catalog.md` (Detailed feature documentation)
  - `tests/test_feature_engineering.py` (25 automated unit tests passing)
- **PLANNED**:
  - Training baseline ML classifiers (Phase 4).
  - Word2Vec/FastText dense embeddings (Phase 5).
  - DeBERTa / RoBERTa fine-tuning pipelines (Phase 5/6).
- **TO BE EXPERIMENTALLY VALIDATED**:
  - Optimal TF-IDF feature vocabulary size ($5,000$ vs $10,000$ vs $20,000$).
  - Effectiveness of synthetic minority oversampling (SMOTE) on tabular vs text features.

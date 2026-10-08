# Feature Catalog & Metadata Registry
## AI-Powered Fake Job and Recruitment Scam Detection System — Phase 3

This catalog documents all engineered features across the 8 functional feature groups. Every feature adheres strictly to zero data leakage protocols and is designed for modular ablation studies.

| Feature Name / Pattern | Group | Approx Dim | Data Source | Calculation | Purpose | Target Leakage Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tfidf_{unigram|bigram}` | **TEXT_TFIDF** | 10000 | clean_composite_text (title, company_profile, description, requirements, benefits) | Sublinear Term Frequency with Inverse Document Frequency and L2 normalization. | Capture high-dimensional lexical keywords, n-gram phrases, and domain-specific vocabulary. | Zero risk. Vocabulary and IDF parameters are fitted strictly on the training partition. |
| `{field}_char_count (title, company_profile, description, requirements, benefits)` | **TEXT_STATISTICS** | 5 | Raw text fields | String length: len(field_text) | Quantify verbosity and completeness of individual posting sections. | Zero risk. Row-level deterministic calculation. |
| `{field}_word_count (title, company_profile, description, requirements, benefits)` | **TEXT_STATISTICS** | 5 | Raw text fields | Word token count: len(field_text.split()) | Measure content volume per section independently of character formatting. | Zero risk. Row-level deterministic calculation. |
| `{field}_avg_word_length (title, company_profile, description, requirements, benefits)` | **TEXT_STATISTICS** | 5 | Raw text fields | char_count / max(word_count, 1) | Capture lexical complexity and presence of concatenated strings. | Zero risk. Row-level deterministic calculation. |
| `{field}_exclamation_count & {field}_question_count` | **TEXT_STATISTICS** | 10 | Raw text fields | field_text.count('!') and field_text.count('?') | Identify sensationalist, urgent, or high-pressure recruitment phrasing. | Zero risk. Row-level deterministic calculation. |
| `{field}_is_present` | **TEXT_STATISTICS** | 5 | Raw text fields | 1 if len(field_text.strip()) > 0 else 0 | Explicit boolean presence indicator for each text section. | Zero risk. Row-level deterministic calculation. |
| `composite_char_count, composite_word_count, composite_avg_word_length` | **TEXT_STATISTICS** | 3 | clean_composite_text | Overall character length, total word count, and average word length. | Measure total posting volume and thoroughness. | Zero risk. Row-level deterministic calculation. |
| `composite_uppercase_count & composite_uppercase_ratio` | **TEXT_STATISTICS** | 2 | clean_composite_text | Uppercase letter count and ratio to total characters. | Detect 'ALL-CAPS' shouting, aggressive capitalization, and scam headlines. | Zero risk. Row-level deterministic calculation. |
| `composite_digit_count & composite_digit_ratio` | **TEXT_STATISTICS** | 2 | clean_composite_text | Count of numeric digits [0-9] and ratio to total characters. | Capture prevalence of numeric payouts, phone digits, and dollar amounts. | Zero risk. Row-level deterministic calculation. |
| `composite_punctuation_count, composite_exclamation_count, composite_question_count` | **TEXT_STATISTICS** | 3 | clean_composite_text | Punctuation frequencies across entire posting. | Capture stylistic punctuation density. | Zero risk. Row-level deterministic calculation. |
| `populated_sections_count` | **TEXT_STATISTICS** | 1 | All 5 text fields | Sum of is_present across (title, company_profile, description, requirements, benefits) [0 to 5]. | Discriminate thorough, legitimate postings from sparse fraudulent listings. | Zero risk. Row-level deterministic calculation. |
| `desc_to_req_length_ratio, desc_to_req_word_ratio, company_profile_to_desc_ratio` | **TEXT_STATISTICS** | 3 | Text section lengths | Laplace-smoothed ratio: (len_A + 1) / (len_B + 1) | Capture structural balance between job overview, requirements, and corporate profile. | Zero risk. Row-level deterministic calculation. |
| `telecommuting, has_company_logo, has_questions` | **METADATA_BINARY** | 3 | telecommuting, has_company_logo, has_questions columns | Coerced to binary integer (0 or 1). | Direct platform signals (e.g. logo absence strongly correlates with fake company accounts). | Zero risk. Tabular source values. |
| `cat_{column}_{category}` | **METADATA_CATEGORICAL** | 250 | employment_type, required_experience, required_education, industry, function, country | One-Hot Encoded with explicit '__MISSING__' category and handle_unknown='ignore'. | Capture structural biases in industry domains, job functions, and country origins. | Zero risk. OneHotEncoder fitted strictly on training partition. |
| `has_company_profile, has_requirements, has_benefits, has_department, has_salary_specified, has_location_specified` | **MISSINGNESS** | 6 | Raw table columns | 1 if column value is non-empty/non-null else 0 | Quantify information omission patterns characteristic of fraudulent campaigns. | Zero risk. Calculated independently of target label. |
| `has_salary_specified, salary_min, salary_max, salary_spread, is_hourly_scale, is_unrealistic_salary` | **SALARY** | 6 | salary_range column | Regex parsing of numeric range 'min-max'; spread = max - min; hourly flag if max <= 250. | Capture compensation availability, wage scale, and unrealistic pay claims. | Zero risk. Deterministic fallback parsing. |
| `email_count, has_email, free_email_count, has_free_email, url_count, has_url, phone_count, has_phone` | **CONTACT** | 8 | clean_composite_text | Compiled regex matching for emails, free webmail domains (@gmail/@yahoo), URLs, and phone numbers. | Detect off-platform communication channels and suspicious free webmail addresses in job ads. | Zero risk. Deterministic regex extraction. |
| `{category}_count, {category}_presence, {category}_freq (for 8 scam categories)` | **HEURISTIC** | 24 | clean_composite_text and config/heuristic_keywords.yaml | Regex boundary matching of categorized scam phrases, binary flag, and word-frequency normalization. | Explicitly represent payment solicitation, urgency, fake guarantees, and crypto terms. | Zero risk. Configured externally without fitting on target. |
| `total_scam_keyword_count, has_any_scam_keyword, scam_category_diversity` | **HEURISTIC** | 3 | Aggregated heuristic outputs | Sum of all scam keyword hits, binary flag, and count of distinct triggered categories. | High-level risk index summarizing overall scam language density. | Zero risk. Purely derived from heuristic keyword counts. |

---

## Detailed Feature Group Breakdown

### Feature Pattern: `tfidf_{unigram|bigram}`
- **Group**: `TEXT_TFIDF`
- **Approximate Dimensions**: `10000`
- **Data Source**: clean_composite_text (title, company_profile, description, requirements, benefits)
- **Mathematical Calculation**: Sublinear Term Frequency with Inverse Document Frequency and L2 normalization.
- **Engineering Purpose**: Capture high-dimensional lexical keywords, n-gram phrases, and domain-specific vocabulary.
- **Potential Limitations**: High dimensionality, sparse representation, susceptible to novel out-of-vocabulary terms.
- **Target Leakage Safeguard**: Zero risk. Vocabulary and IDF parameters are fitted strictly on the training partition.

### Feature Pattern: `{field}_char_count (title, company_profile, description, requirements, benefits)`
- **Group**: `TEXT_STATISTICS`
- **Approximate Dimensions**: `5`
- **Data Source**: Raw text fields
- **Mathematical Calculation**: String length: len(field_text)
- **Engineering Purpose**: Quantify verbosity and completeness of individual posting sections.
- **Potential Limitations**: Does not assess semantic quality of text.
- **Target Leakage Safeguard**: Zero risk. Row-level deterministic calculation.

### Feature Pattern: `{field}_word_count (title, company_profile, description, requirements, benefits)`
- **Group**: `TEXT_STATISTICS`
- **Approximate Dimensions**: `5`
- **Data Source**: Raw text fields
- **Mathematical Calculation**: Word token count: len(field_text.split())
- **Engineering Purpose**: Measure content volume per section independently of character formatting.
- **Potential Limitations**: Language-agnostic whitespace splitting.
- **Target Leakage Safeguard**: Zero risk. Row-level deterministic calculation.

### Feature Pattern: `{field}_avg_word_length (title, company_profile, description, requirements, benefits)`
- **Group**: `TEXT_STATISTICS`
- **Approximate Dimensions**: `5`
- **Data Source**: Raw text fields
- **Mathematical Calculation**: char_count / max(word_count, 1)
- **Engineering Purpose**: Capture lexical complexity and presence of concatenated strings.
- **Potential Limitations**: Sensitive to long URLs or non-spaced tokens.
- **Target Leakage Safeguard**: Zero risk. Row-level deterministic calculation.

### Feature Pattern: `{field}_exclamation_count & {field}_question_count`
- **Group**: `TEXT_STATISTICS`
- **Approximate Dimensions**: `10`
- **Data Source**: Raw text fields
- **Mathematical Calculation**: field_text.count('!') and field_text.count('?')
- **Engineering Purpose**: Identify sensationalist, urgent, or high-pressure recruitment phrasing.
- **Potential Limitations**: Some legitimate postings occasionally use exclamation marks.
- **Target Leakage Safeguard**: Zero risk. Row-level deterministic calculation.

### Feature Pattern: `{field}_is_present`
- **Group**: `TEXT_STATISTICS`
- **Approximate Dimensions**: `5`
- **Data Source**: Raw text fields
- **Mathematical Calculation**: 1 if len(field_text.strip()) > 0 else 0
- **Engineering Purpose**: Explicit boolean presence indicator for each text section.
- **Potential Limitations**: Binary only; does not measure length.
- **Target Leakage Safeguard**: Zero risk. Row-level deterministic calculation.

### Feature Pattern: `composite_char_count, composite_word_count, composite_avg_word_length`
- **Group**: `TEXT_STATISTICS`
- **Approximate Dimensions**: `3`
- **Data Source**: clean_composite_text
- **Mathematical Calculation**: Overall character length, total word count, and average word length.
- **Engineering Purpose**: Measure total posting volume and thoroughness.
- **Potential Limitations**: Aggregates across all sections.
- **Target Leakage Safeguard**: Zero risk. Row-level deterministic calculation.

### Feature Pattern: `composite_uppercase_count & composite_uppercase_ratio`
- **Group**: `TEXT_STATISTICS`
- **Approximate Dimensions**: `2`
- **Data Source**: clean_composite_text
- **Mathematical Calculation**: Uppercase letter count and ratio to total characters.
- **Engineering Purpose**: Detect 'ALL-CAPS' shouting, aggressive capitalization, and scam headlines.
- **Potential Limitations**: Legitimate acronyms contribute slightly to uppercase count.
- **Target Leakage Safeguard**: Zero risk. Row-level deterministic calculation.

### Feature Pattern: `composite_digit_count & composite_digit_ratio`
- **Group**: `TEXT_STATISTICS`
- **Approximate Dimensions**: `2`
- **Data Source**: clean_composite_text
- **Mathematical Calculation**: Count of numeric digits [0-9] and ratio to total characters.
- **Engineering Purpose**: Capture prevalence of numeric payouts, phone digits, and dollar amounts.
- **Potential Limitations**: Legitimate postings contain dates, phone numbers, and compensation.
- **Target Leakage Safeguard**: Zero risk. Row-level deterministic calculation.

### Feature Pattern: `composite_punctuation_count, composite_exclamation_count, composite_question_count`
- **Group**: `TEXT_STATISTICS`
- **Approximate Dimensions**: `3`
- **Data Source**: clean_composite_text
- **Mathematical Calculation**: Punctuation frequencies across entire posting.
- **Engineering Purpose**: Capture stylistic punctuation density.
- **Potential Limitations**: Standard grammatical punctuation included.
- **Target Leakage Safeguard**: Zero risk. Row-level deterministic calculation.

### Feature Pattern: `populated_sections_count`
- **Group**: `TEXT_STATISTICS`
- **Approximate Dimensions**: `1`
- **Data Source**: All 5 text fields
- **Mathematical Calculation**: Sum of is_present across (title, company_profile, description, requirements, benefits) [0 to 5].
- **Engineering Purpose**: Discriminate thorough, legitimate postings from sparse fraudulent listings.
- **Potential Limitations**: Some legitimate rapid job listings may omit benefits.
- **Target Leakage Safeguard**: Zero risk. Row-level deterministic calculation.

### Feature Pattern: `desc_to_req_length_ratio, desc_to_req_word_ratio, company_profile_to_desc_ratio`
- **Group**: `TEXT_STATISTICS`
- **Approximate Dimensions**: `3`
- **Data Source**: Text section lengths
- **Mathematical Calculation**: Laplace-smoothed ratio: (len_A + 1) / (len_B + 1)
- **Engineering Purpose**: Capture structural balance between job overview, requirements, and corporate profile.
- **Potential Limitations**: Requires safe smoothing to avoid division by zero.
- **Target Leakage Safeguard**: Zero risk. Row-level deterministic calculation.

### Feature Pattern: `telecommuting, has_company_logo, has_questions`
- **Group**: `METADATA_BINARY`
- **Approximate Dimensions**: `3`
- **Data Source**: telecommuting, has_company_logo, has_questions columns
- **Mathematical Calculation**: Coerced to binary integer (0 or 1).
- **Engineering Purpose**: Direct platform signals (e.g. logo absence strongly correlates with fake company accounts).
- **Potential Limitations**: Early-stage legitimate startups may lack logos or screening questions.
- **Target Leakage Safeguard**: Zero risk. Tabular source values.

### Feature Pattern: `cat_{column}_{category}`
- **Group**: `METADATA_CATEGORICAL`
- **Approximate Dimensions**: `250`
- **Data Source**: employment_type, required_experience, required_education, industry, function, country
- **Mathematical Calculation**: One-Hot Encoded with explicit '__MISSING__' category and handle_unknown='ignore'.
- **Engineering Purpose**: Capture structural biases in industry domains, job functions, and country origins.
- **Potential Limitations**: Increases matrix dimensionality; sparse categories with low frequency.
- **Target Leakage Safeguard**: Zero risk. OneHotEncoder fitted strictly on training partition.

### Feature Pattern: `has_company_profile, has_requirements, has_benefits, has_department, has_salary_specified, has_location_specified`
- **Group**: `MISSINGNESS`
- **Approximate Dimensions**: `6`
- **Data Source**: Raw table columns
- **Mathematical Calculation**: 1 if column value is non-empty/non-null else 0
- **Engineering Purpose**: Quantify information omission patterns characteristic of fraudulent campaigns.
- **Potential Limitations**: Some small employers do not provide departments or benefits.
- **Target Leakage Safeguard**: Zero risk. Calculated independently of target label.

### Feature Pattern: `has_salary_specified, salary_min, salary_max, salary_spread, is_hourly_scale, is_unrealistic_salary`
- **Group**: `SALARY`
- **Approximate Dimensions**: `6`
- **Data Source**: salary_range column
- **Mathematical Calculation**: Regex parsing of numeric range 'min-max'; spread = max - min; hourly flag if max <= 250.
- **Engineering Purpose**: Capture compensation availability, wage scale, and unrealistic pay claims.
- **Potential Limitations**: Salary is missing in ~83% of entries; unparseable strings default to 0.
- **Target Leakage Safeguard**: Zero risk. Deterministic fallback parsing.

### Feature Pattern: `email_count, has_email, free_email_count, has_free_email, url_count, has_url, phone_count, has_phone`
- **Group**: `CONTACT`
- **Approximate Dimensions**: `8`
- **Data Source**: clean_composite_text
- **Mathematical Calculation**: Compiled regex matching for emails, free webmail domains (@gmail/@yahoo), URLs, and phone numbers.
- **Engineering Purpose**: Detect off-platform communication channels and suspicious free webmail addresses in job ads.
- **Potential Limitations**: Legitimate freelancers or small shops occasionally use Gmail.
- **Target Leakage Safeguard**: Zero risk. Deterministic regex extraction.

### Feature Pattern: `{category}_count, {category}_presence, {category}_freq (for 8 scam categories)`
- **Group**: `HEURISTIC`
- **Approximate Dimensions**: `24`
- **Data Source**: clean_composite_text and config/heuristic_keywords.yaml
- **Mathematical Calculation**: Regex boundary matching of categorized scam phrases, binary flag, and word-frequency normalization.
- **Engineering Purpose**: Explicitly represent payment solicitation, urgency, fake guarantees, and crypto terms.
- **Potential Limitations**: Adversarial scammers may alter phrase wording to evade static lists.
- **Target Leakage Safeguard**: Zero risk. Configured externally without fitting on target.

### Feature Pattern: `total_scam_keyword_count, has_any_scam_keyword, scam_category_diversity`
- **Group**: `HEURISTIC`
- **Approximate Dimensions**: `3`
- **Data Source**: Aggregated heuristic outputs
- **Mathematical Calculation**: Sum of all scam keyword hits, binary flag, and count of distinct triggered categories.
- **Engineering Purpose**: High-level risk index summarizing overall scam language density.
- **Potential Limitations**: Composite heuristic score; requires combination with ML models.
- **Target Leakage Safeguard**: Zero risk. Purely derived from heuristic keyword counts.

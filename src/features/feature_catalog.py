"""
Comprehensive Feature Catalog Registry and Documentation Generator.

Maintains formal metadata, mathematical definitions, data sources, rationale,
potential limitations, and leakage risks for every engineered feature in the system.
"""

from typing import Dict, List, Any
import json
import os


FEATURE_CATALOG_REGISTRY: List[Dict[str, Any]] = [
    # 1. TEXT_TFIDF
    {
        "group": "TEXT_TFIDF",
        "name_pattern": "tfidf_{unigram|bigram}",
        "dimension_approx": 10000,
        "source": "clean_composite_text (title, company_profile, description, requirements, benefits)",
        "calculation": "Sublinear Term Frequency with Inverse Document Frequency and L2 normalization.",
        "purpose": "Capture high-dimensional lexical keywords, n-gram phrases, and domain-specific vocabulary.",
        "limitations": "High dimensionality, sparse representation, susceptible to novel out-of-vocabulary terms.",
        "leakage_risk": "Zero risk. Vocabulary and IDF parameters are fitted strictly on the training partition.",
    },

    # 2. TEXT_STATISTICS (Per Field)
    {
        "group": "TEXT_STATISTICS",
        "name_pattern": "{field}_char_count (title, company_profile, description, requirements, benefits)",
        "dimension_approx": 5,
        "source": "Raw text fields",
        "calculation": "String length: len(field_text)",
        "purpose": "Quantify verbosity and completeness of individual posting sections.",
        "limitations": "Does not assess semantic quality of text.",
        "leakage_risk": "Zero risk. Row-level deterministic calculation.",
    },
    {
        "group": "TEXT_STATISTICS",
        "name_pattern": "{field}_word_count (title, company_profile, description, requirements, benefits)",
        "dimension_approx": 5,
        "source": "Raw text fields",
        "calculation": "Word token count: len(field_text.split())",
        "purpose": "Measure content volume per section independently of character formatting.",
        "limitations": "Language-agnostic whitespace splitting.",
        "leakage_risk": "Zero risk. Row-level deterministic calculation.",
    },
    {
        "group": "TEXT_STATISTICS",
        "name_pattern": "{field}_avg_word_length (title, company_profile, description, requirements, benefits)",
        "dimension_approx": 5,
        "source": "Raw text fields",
        "calculation": "char_count / max(word_count, 1)",
        "purpose": "Capture lexical complexity and presence of concatenated strings.",
        "limitations": "Sensitive to long URLs or non-spaced tokens.",
        "leakage_risk": "Zero risk. Row-level deterministic calculation.",
    },
    {
        "group": "TEXT_STATISTICS",
        "name_pattern": "{field}_exclamation_count & {field}_question_count",
        "dimension_approx": 10,
        "source": "Raw text fields",
        "calculation": "field_text.count('!') and field_text.count('?')",
        "purpose": "Identify sensationalist, urgent, or high-pressure recruitment phrasing.",
        "limitations": "Some legitimate postings occasionally use exclamation marks.",
        "leakage_risk": "Zero risk. Row-level deterministic calculation.",
    },
    {
        "group": "TEXT_STATISTICS",
        "name_pattern": "{field}_is_present",
        "dimension_approx": 5,
        "source": "Raw text fields",
        "calculation": "1 if len(field_text.strip()) > 0 else 0",
        "purpose": "Explicit boolean presence indicator for each text section.",
        "limitations": "Binary only; does not measure length.",
        "leakage_risk": "Zero risk. Row-level deterministic calculation.",
    },

    # 2. TEXT_STATISTICS (Composite & Ratios)
    {
        "group": "TEXT_STATISTICS",
        "name_pattern": "composite_char_count, composite_word_count, composite_avg_word_length",
        "dimension_approx": 3,
        "source": "clean_composite_text",
        "calculation": "Overall character length, total word count, and average word length.",
        "purpose": "Measure total posting volume and thoroughness.",
        "limitations": "Aggregates across all sections.",
        "leakage_risk": "Zero risk. Row-level deterministic calculation.",
    },
    {
        "group": "TEXT_STATISTICS",
        "name_pattern": "composite_uppercase_count & composite_uppercase_ratio",
        "dimension_approx": 2,
        "source": "clean_composite_text",
        "calculation": "Uppercase letter count and ratio to total characters.",
        "purpose": "Detect 'ALL-CAPS' shouting, aggressive capitalization, and scam headlines.",
        "limitations": "Legitimate acronyms contribute slightly to uppercase count.",
        "leakage_risk": "Zero risk. Row-level deterministic calculation.",
    },
    {
        "group": "TEXT_STATISTICS",
        "name_pattern": "composite_digit_count & composite_digit_ratio",
        "dimension_approx": 2,
        "source": "clean_composite_text",
        "calculation": "Count of numeric digits [0-9] and ratio to total characters.",
        "purpose": "Capture prevalence of numeric payouts, phone digits, and dollar amounts.",
        "limitations": "Legitimate postings contain dates, phone numbers, and compensation.",
        "leakage_risk": "Zero risk. Row-level deterministic calculation.",
    },
    {
        "group": "TEXT_STATISTICS",
        "name_pattern": "composite_punctuation_count, composite_exclamation_count, composite_question_count",
        "dimension_approx": 3,
        "source": "clean_composite_text",
        "calculation": "Punctuation frequencies across entire posting.",
        "purpose": "Capture stylistic punctuation density.",
        "limitations": "Standard grammatical punctuation included.",
        "leakage_risk": "Zero risk. Row-level deterministic calculation.",
    },
    {
        "group": "TEXT_STATISTICS",
        "name_pattern": "populated_sections_count",
        "dimension_approx": 1,
        "source": "All 5 text fields",
        "calculation": "Sum of is_present across (title, company_profile, description, requirements, benefits) [0 to 5].",
        "purpose": "Discriminate thorough, legitimate postings from sparse fraudulent listings.",
        "limitations": "Some legitimate rapid job listings may omit benefits.",
        "leakage_risk": "Zero risk. Row-level deterministic calculation.",
    },
    {
        "group": "TEXT_STATISTICS",
        "name_pattern": "desc_to_req_length_ratio, desc_to_req_word_ratio, company_profile_to_desc_ratio",
        "dimension_approx": 3,
        "source": "Text section lengths",
        "calculation": "Laplace-smoothed ratio: (len_A + 1) / (len_B + 1)",
        "purpose": "Capture structural balance between job overview, requirements, and corporate profile.",
        "limitations": "Requires safe smoothing to avoid division by zero.",
        "leakage_risk": "Zero risk. Row-level deterministic calculation.",
    },

    # 3. METADATA_BINARY
    {
        "group": "METADATA_BINARY",
        "name_pattern": "telecommuting, has_company_logo, has_questions",
        "dimension_approx": 3,
        "source": "telecommuting, has_company_logo, has_questions columns",
        "calculation": "Coerced to binary integer (0 or 1).",
        "purpose": "Direct platform signals (e.g. logo absence strongly correlates with fake company accounts).",
        "limitations": "Early-stage legitimate startups may lack logos or screening questions.",
        "leakage_risk": "Zero risk. Tabular source values.",
    },

    # 4. METADATA_CATEGORICAL
    {
        "group": "METADATA_CATEGORICAL",
        "name_pattern": "cat_{column}_{category}",
        "dimension_approx": 250,
        "source": "employment_type, required_experience, required_education, industry, function, country",
        "calculation": "One-Hot Encoded with explicit '__MISSING__' category and handle_unknown='ignore'.",
        "purpose": "Capture structural biases in industry domains, job functions, and country origins.",
        "limitations": "Increases matrix dimensionality; sparse categories with low frequency.",
        "leakage_risk": "Zero risk. OneHotEncoder fitted strictly on training partition.",
    },

    # 5. MISSINGNESS
    {
        "group": "MISSINGNESS",
        "name_pattern": "has_company_profile, has_requirements, has_benefits, has_department, has_salary_specified, has_location_specified",
        "dimension_approx": 6,
        "source": "Raw table columns",
        "calculation": "1 if column value is non-empty/non-null else 0",
        "purpose": "Quantify information omission patterns characteristic of fraudulent campaigns.",
        "limitations": "Some small employers do not provide departments or benefits.",
        "leakage_risk": "Zero risk. Calculated independently of target label.",
    },

    # 6. SALARY
    {
        "group": "SALARY",
        "name_pattern": "has_salary_specified, salary_min, salary_max, salary_spread, is_hourly_scale, is_unrealistic_salary",
        "dimension_approx": 6,
        "source": "salary_range column",
        "calculation": "Regex parsing of numeric range 'min-max'; spread = max - min; hourly flag if max <= 250.",
        "purpose": "Capture compensation availability, wage scale, and unrealistic pay claims.",
        "limitations": "Salary is missing in ~83% of entries; unparseable strings default to 0.",
        "leakage_risk": "Zero risk. Deterministic fallback parsing.",
    },

    # 7. CONTACT
    {
        "group": "CONTACT",
        "name_pattern": "email_count, has_email, free_email_count, has_free_email, url_count, has_url, phone_count, has_phone",
        "dimension_approx": 8,
        "source": "clean_composite_text",
        "calculation": "Compiled regex matching for emails, free webmail domains (@gmail/@yahoo), URLs, and phone numbers.",
        "purpose": "Detect off-platform communication channels and suspicious free webmail addresses in job ads.",
        "limitations": "Legitimate freelancers or small shops occasionally use Gmail.",
        "leakage_risk": "Zero risk. Deterministic regex extraction.",
    },

    # 8. HEURISTIC
    {
        "group": "HEURISTIC",
        "name_pattern": "{category}_count, {category}_presence, {category}_freq (for 8 scam categories)",
        "dimension_approx": 24,
        "source": "clean_composite_text and config/heuristic_keywords.yaml",
        "calculation": "Regex boundary matching of categorized scam phrases, binary flag, and word-frequency normalization.",
        "purpose": "Explicitly represent payment solicitation, urgency, fake guarantees, and crypto terms.",
        "limitations": "Adversarial scammers may alter phrase wording to evade static lists.",
        "leakage_risk": "Zero risk. Configured externally without fitting on target.",
    },
    {
        "group": "HEURISTIC",
        "name_pattern": "total_scam_keyword_count, has_any_scam_keyword, scam_category_diversity",
        "dimension_approx": 3,
        "source": "Aggregated heuristic outputs",
        "calculation": "Sum of all scam keyword hits, binary flag, and count of distinct triggered categories.",
        "purpose": "High-level risk index summarizing overall scam language density.",
        "limitations": "Composite heuristic score; requires combination with ML models.",
        "leakage_risk": "Zero risk. Purely derived from heuristic keyword counts.",
    },
]


def generate_markdown_catalog() -> str:
    """Generates a complete markdown report from the feature catalog registry."""
    md = [
        "# Feature Catalog & Metadata Registry",
        "## AI-Powered Fake Job and Recruitment Scam Detection System — Phase 3\n",
        "This catalog documents all engineered features across the 8 functional feature groups. "
        "Every feature adheres strictly to zero data leakage protocols and is designed for modular ablation studies.\n",
        "| Feature Name / Pattern | Group | Approx Dim | Data Source | Calculation | Purpose | Target Leakage Risk |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for item in FEATURE_CATALOG_REGISTRY:
        md.append(
            f"| `{item['name_pattern']}` | **{item['group']}** | {item['dimension_approx']} | {item['source']} | "
            f"{item['calculation']} | {item['purpose']} | {item['leakage_risk']} |"
        )

    md.extend([
        "\n---\n",
        "## Detailed Feature Group Breakdown\n",
    ])

    for item in FEATURE_CATALOG_REGISTRY:
        md.extend([
            f"### Feature Pattern: `{item['name_pattern']}`",
            f"- **Group**: `{item['group']}`",
            f"- **Approximate Dimensions**: `{item['dimension_approx']}`",
            f"- **Data Source**: {item['source']}",
            f"- **Mathematical Calculation**: {item['calculation']}",
            f"- **Engineering Purpose**: {item['purpose']}",
            f"- **Potential Limitations**: {item['limitations']}",
            f"- **Target Leakage Safeguard**: {item['leakage_risk']}",
            "",
        ])

    return "\n".join(md)

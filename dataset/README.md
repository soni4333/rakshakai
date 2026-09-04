# RakshakAI Dataset Documentation

This directory contains the multi-layer training and evaluation datasets for **RakshakAI**, designed to detect unfair contract terms, regulatory non-compliance, and deceptive clauses under Indian legal frameworks (DPDP Act 2023, RBI Digital Lending Guidelines 2022/2025, Code on Wages 2019, State Gig Worker Welfare Acts, etc.).

---

## Dataset Architecture

The dataset is constructed across **4 distinct layers**:

| Layer | Source / Purpose | Row Count | Labeling Method | Needs Review Flag |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1** (`layer1_claudette/`) | Hugging Face (`mteb/UnfairTOSLegalBenchClassification`). Pretraining benchmark for general unfair Terms of Service. | ~2,057 | Pre-labeled (CLAUDETTE) | `false` |
| **Layer 2** (`layer2_real/`) | Ground-truth real-world Indian gig-work and fintech violation clauses (`real_clauses.xlsx`). | 93 | Expert Ground Truth | `false` |
| **Layer 3** (`layer3_synthetic/`) | Synthetic violating & compliant clause pairs generated for DPDP Act 2023 & RBI Digital Lending Guidelines. | 120 | Labeled by construction | `false` |
| **Layer 4** (`layer4_adversarial/`) | Disguised-phrasing adversarial rewrites (evasive legal wording, passive voice, double negatives). | 48 | Adversarial Rewrites | `true` |

---

## Schema Definition

All datasets are normalized into a unified schema when merged:

- `id`: Unique example identifier (e.g. `L1_1`, `L2_15`, `L3_42`, `L4_10`).
- `layer`: Layer number (`1`, `2`, `3`, or `4`).
- `category`: Domain category (`Gig Contract`, `Loan App / Fintech`, `General Terms of Service`).
- `text`: Clause text or finding description.
- `violation_type`: Specific violation category (e.g. `Unilateral financial penalty`, `Prohibited contact access`, `Unconsented data sharing`).
- `law_statute`: Relevant law or provision (e.g. `DPDP Act 2023 §5`, `RBI Digital Lending Guidelines 2022`, `Indian Contract Act 1872`).
- `label`: Risk classification (`High-risk`, `Medium-risk`, or `Low-risk`).
- `source_name`: Source organization or document name.
- `source_url`: Verifiable citation URL.
- `notes`: Auditor notes or context.
- `needs_human_review`: Boolean flag (`true` for Layer 4 adversarial rows requiring manual human review).

---

## How to Add More Layer 2 Real Examples

1. Open `dataset/layer2_real/real_clauses.xlsx` in Excel or pandas.
2. Add your new row with the following columns:
   - `ID`: Next numeric integer (e.g. `95`, `96`, ...).
   - `Category`: `Gig Contract` or `Loan App / Fintech`.
   - `Clause / Finding Description`: Text of the clause or finding.
   - `Violation Type`: Specific non-compliance type.
   - `Relevant Law / Statute`: Specific section/act.
   - `Label`: `High-risk`, `Medium-risk`, or `Low-risk`.
   - `Source Name`: Source publication or document name.
   - `Source URL`: Direct web URL or PDF citation.
   - `Notes`: Additional reviewer notes.
3. Save the updated `real_clauses.xlsx` file.
4. Run the rebuild script from the root directory:
   ```bash
   python scripts/rebuild_dataset.py
   ```
5. The pipeline will automatically reload Layer 2, rebuild the merged dataset, and regenerate the `70 / 15 / 15` train/validation/test splits in `dataset/merged_training_set/`.

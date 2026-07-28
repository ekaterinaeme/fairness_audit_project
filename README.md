# Credit Risk Model Fairness Audit

A compliance-focused fairness audit and decision engine optimization for a logistic regression credit risk model, aligned with EU AI Act (Annex III 5(b) - creditworthiness assessment).

## Purpose

The EU AI Act imposes strict requirements on high-risk AI systems, including creditworthiness assessments. This audit demonstrates a practical methodology for:
- Detecting disparate impact in production models
- Remediating bias through decision threshold and logit shift optimization
- Documenting findings in a regulator‑ready format
- Identifying gaps that require further investigation

## Methodology
Three-tier remediation framework:
1. **Pre-processing** — protected attributes excluded from model features
2. **Post-processing** — group-aware threshold/logit adjustment (implemented; disclosed as a temporary remediation, not a compliance conclusion)
3. **In-processing** — fairness-constrained training, e.g. Fairlearn's ExponentiatedGradient (recommended, not implemented)


## Key Findings

| **Metric** | **Baseline** | **Remediated** | **Test Set** | **Status** |
|------------|--------------|----------------|--------------|------------|
| Disparate Impact (DI) | 0.60 | 0.84 | 1.12 (CI: 0.58–1.85) | Point estimate compliant |
| FPR Parity | 1.25 | 1.02 | 0.69 (CI: 0.32–1.16) | Fails |
| TPR Parity | 1.13 | 0.96 | 1.03 (CI: 0.77–1.29) | Compliant |
| Financial Loss | €57,663 | €52,423 | €63,545 | — |
| Manual Review Routing (Age) | — | — | 1.38× | Gap identified |

## Regulatory Scope

- **Framework:** EU AI Act (High-risk Classification, Annex III 5 (b))
- **Protected Attribute:** Age (under 25)
- **Fairness Metrics:** Disparate Impact, Equalized Odds (with bootstrapped confidence intervals), Proxy Discrimination AUC
- **Outputs:** Executive Summary, JSON Audit Trail, Fairness-Cost Frontier

## Repository Structure
```
├── README.md # This file
├── executive_summary.pdf # Compliance report in PDF
├── fairness_audit.py # Main Python script
├── audit_record.json # Structured audit trail
├── fairness_cost_frontier.png # Pareto frontier visualization
├── german.data # Dataset
```
## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib

# Run the Audit (generates console output, fairness_cost_frontier.png, and audit_record.json) 
python fairness_audit.py
```
**Note:** The German Credit dataset (`german.data`) must be placed in the same directory

## Key Artifacts

| **File** | **Description** |
|----------|-----------------|
| `fairness_audit.py` | Full pipeline: preprocessing, model training, grid search over threshold and logit shift, decision routing with manual review zone, audit generation, and visualisation. |
| `audit_record.json` | Structured JSON audit trail containing all metrics, hyperparameters, dataset splits, cost assumptions, and sensitivity results—ready for regulatory submission. |
| `fairness_cost_frontier.png` | Visual Pareto frontier showing the fairness-cost tradeoff across protected attributes (grid searched on age only; sex/foreign_worker are shown as cross-attribute effect). |
| `executive_summary.pdf` | One-page compliance report summarising findings, remediation, unresolved issues, and recommendations. |

## Limitations

- Test set: 200 samples; wide bootstrapped confidence intervals
- `personal_status` conflates sex and marital status; single-female category contains too few observations
- Foreign-worker attribute: reference group too small (n=6, test set) for reliable DI assessment
- **This audit does not conclude full regulatory compliance readiness** — see Recommendation in `executive_summary.pdf`

# Credit Risk Model Fairness Audit

**Date:** July 2026
**Author:** Ekaterina Emelianova
**Scope:** Logistic Regression Model for credit risk, evaluated under EU AI Act (Annex III, 5(b) – Creditworthiness Assessment)
**Protected Attribute:** Age (< 25)

## Executive Summary

This audit evaluates a logistic regression model for credit risk under the EU AI Act's high-risk classification. The protected attribute is age, with applicants under 25 designated as the protected group.

**Key Findings:** The baseline model exhibited disparate impact against the protected group, approving credit applications at a 40% lower rate than the reference group (age ≥ 25). This corresponds to a Disparate Impact (DI) ratio of 0.60, violating the 80% rule—a widely adopted benchmark in financial fairness auditing.

**Remediation:** A bounded grid search over decision threshold and logit shift was applied. The selected configuration achieved a DI ratio of 0.84 on the validation set, which falls within the 0.80–1.25 compliance band, and reduced the estimated financial loss by €5,240 (from €57,663 to €52,423).

**Final Evaluation:** The remediated model was evaluated on a held-out test set. The final DI ratio is 1.12, which remains within the compliance band and stable across FN:FP cost ratios from 2:1 to 6:1 (including the estimated 3.4:1 ratio). However, equalized odds fails at lower ratios (1:1 and 1.5:1). While the validation set FPR ratio (1.02) meets the 0.80–1.25 band, the test set FPR ratio (0.69) falls below it. Wide confidence intervals (DI 95% CI: 0.58–1.85) indicate high statistical uncertainty due to the limited test sample (n=200). This finding should be treated as directional, not conclusive, until validated on a larger cohort.

## Secondary Findings

1. **Manual Review Disparity.** Applicants under 25 are routed to manual review at ~1.38× the rate of older applicants, suggesting age-related disparity extends beyond automated decisions and warrants operational review.
2. **Proxy Discrimination Risk.** Remaining features predict foreign worker status with AUC = 0.738, indicating potential indirect bias that persists despite protected attribute exclusion.
3. **Foreign Worker Assessment Limited.** Insufficient native worker representation (n=37 full dataset, n=6 test set) prevents statistically reliable DI evaluation for this group.
4. **Sex Attribute Note.** DI ratio for female applicants is 1.15 (within band). However, the dataset's `personal_status` field conflates sex with marital status, and the single-female category contains too few observations to yield reliable estimates, limiting interpretability.

## Statistical Limitations

The test set contains only 200 samples. Bootstrapped confidence intervals are wide, meaning the compliant point estimate should be treated as directional, not conclusive, until validated on a larger, representative dataset.

## Recommendation

This audit does not conclude full regulatory readiness. While DI compliance for age is achieved at the selected operating point, the following require resolution before production deployment:

- Equalized odds failure on the test set
- Unresolved disparity in manual review escalation
- Proxy discrimination risk regarding nationality
- Statistical uncertainty due to sample size

Before deployment, the audit recommends: (1) validating these findings on a larger, representative dataset; and (2) adopting an in-processing fairness-constrained approach (e.g., Fairlearn's `ExponentiatedGradient`) to embed fairness during training and remove dependency on protected attributes at inference time.

---

*Full technical documentation, reproducible audit code, and JSON audit trail are available in the project repository upon request.*

# Credit Risk Model Fairness Audit

**Date:** July 2026
**Author:** Ekaterina Emelianova
**Scope:** Logistic Regression Model for credit risk under EU AI Act (Annex III, 5(b) - Creditworthiness Assessment)
**Protected Attribute:** Age (< 25)

## Executive Summary

This audit evaluates a logistic regression model for credit applications under the EU AI Act's high-risk classification. It examines how well the model performs regarding applicants aged under 25, focusing on its fairness.

**Key Findings:** The initial model approved younger applicants at a 40% lower rate than those over 25, which raised concerns about bias. The Disparate Impact (DI) ratio of 0.60 violated the 80% rule, which is a standard benchmark in financial fairness auditing.

**Remediation:** To address this, a bounded grid search was applied over decision threshold and logit shift. These changes improved the model's fairness-cost trade-off, achieving a DI ratio of 0.84 on the validation set (which is within the 0.80–1.25 compliance band), and reducing financial loss by €5,240 (from €57,663 to €52,423).

**Final Evaluation:** Testing on a held-out dataset revealed that the revised model achieved a DI ratio of 1.12, staying within the compliance band and being stable across FN:FP cost ratios from 2:1 to 6:1 (including the estimated 3.4:1 ratio). However, at lower ratios like 1:1 and 1.5:1, the model struggled to meet fairness benchmarks. While validation data FPR ratio showed acceptable results (1.02, which is within the 0.80-1.25 compliance band), testing data saw the FPR ratio drop to 0.69, indicating potential issues. Wide confidence intervals (DI 95% CI: 0.58–1.85) indicate high statistical uncertainty due to the limited test sample (n=200). This finding should be treated as directional rather than conclusive, until it is validated on a larger dataset.

## Secondary Findings

1. **Manual Review Disparity.** Younger applicants were more likely to face manual reviews at ~1.38× higher rate than older applicants, suggesting age-related disparity beyond automated decisions.
2. **Proxy Discrimination Risk.** Remaining features predict foreign worker status with AUC = 0.738, indicating potential indirect bias that persists despite the exclusion of protected attributes.
3. **Foreign Worker Assessment Limited.** Native workers are insufficiently represented in the dataset (n=37 full dataset, n=6 test set), which makes DI evaluation for this group unreliable.
4. **Sex Attribute Note.** DI ratio for female applicants is 1.15 (within band). However, the dataset's `personal_status` field conflates sex with marital status, and the single-female category contains too few observations for reliable estimates, therefore interpretability is limited.

## Data Limitations

With only 200 samples in the test set, these findings are less robust: bootstrapped confidence intervals are wide.

## Recommendation

This audit does not conclude full regulatory readiness. While DI compliance for age was achieved, the following issues require resolution: 
- Equalized odds fail on the test set
- Disparity issue unresolved in manual review escalation
- Proxy discrimination risk for nationality 
- Small sample size leads to statistical uncertainty

To ensure reliability, deployment should wait until a larger dataset is available and techniques like Fairlearn’s ExponentiatedGradient are integrated during training. This proactive approach will help embed fairness into the model's design, minimizing biased outcomes. 

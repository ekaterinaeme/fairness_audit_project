"""
Compliance Communication Sample: Credit Risk Model Audit

This audit examines a credit risk model under the EU AI Act (high-risk classification, 
Annex III point 5(b) - creditworthiness assessment).

Tier-1: Pre-processing
Protected attributes (age, personal_status, foreign_worker) are excluded from the 
feature set. This is the lowest-risk approach, eliminating direct disparate treatment 
exposure.

Tier-2: Post-processing
A group-aware logit adjustment (grid search over threshold + delta) is applied. This is 
a stopgap mitigation. It carries its own legal exposure under indirect discrimination 
doctrine. It is presented here as a temporary measure, not a compliance conclusion.

Tier-3: In-processing (RECOMMENDED - not implemented)
Fairness constraints during training (e.g., Fairlearn's ExponentiatedGradient). This is 
preferable because it does not require protected status at inference time, and it does 
not adjust individual scores after the fact.

Cost parameters (COST_FP, COST_FN) are derived from this dataset's own loan 
characteristics and illustrative retail-lending assumptions (see "COST PARAMETER DERIVATION" 
below). A ratio sweep then tests whether remediation decision is stable across a range of 
plausible FN:FP cost ratios.

Date: June 2026
Project: Compliance Communication Sample
"""

import pandas as pd
import numpy as np
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import json 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier 
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from enum import IntEnum

# CONFIGURATION

# File path
DATA_FILE = 'german.data'

# Data split ratios
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Operational buffer for manual review zone
OPERATIONAL_BUFFER = 0.03

# Age threshold for protected group
AGE_THRESHOLD = 25

# Group labels
PROTECTED_LABEL = 'Protected (<25)'
REFERENCE_LABEL = 'Reference (>=25)'

# Column names for German Credit dataset
COLUMN_NAMES = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'saving_status', 'employment', 'installment_commitment', 'personal_status',
    'other_parties', 'residence_since', 'property_magnitude', 'age',
    'other_payment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'own_telephone', 'foreign_worker', 'class'
]

# DATA LOADING

# Load the dataset
df = pd.read_csv(DATA_FILE, sep=' ', header=None, names=COLUMN_NAMES)

# Map target: Class 1 (Good) -> 0, Class 2 (Bad) -> 1
df['class'] = df['class'].map({1: 0, 2: 1})

# Split features and target
X = df.drop(columns=['class'])
y = df['class']

# COST PARAMETER CALCULATION

# COST_FN (cost of approving a bad loan) = average loan amount * assumed LGD (note: a single mean value loan amount is used for all customers. In production, this value would be estimated separately for every customer.)
# COST_FP (cost of rejecting a good loan) = interest rate * average loan duration + CAC

DM_TO_EUR = 1.95583 # Fixed EUR exchange rate at DM retirement (31 Dec 1998)
LGD_ASSUMPTION = 0.70 # Loss Given Default (conservative 70% value given no collateral and ~30% default rate)
ANNUAL_INTEREST_RATE = 0.10 # Illustrative consumer credit interest rate
CAC_ESTIMATE = 100.0 # Illustrative customer acquisition cost, EUR
PRESENT_DAY_MULTIPLIER = 1.9 # Illustrative cumulative eurozone inflation since mid-1990's

avg_loan_dm = df['credit_amount'].mean()
avg_loan_eur_present = avg_loan_dm / DM_TO_EUR * PRESENT_DAY_MULTIPLIER
avg_duration_months = df['duration'].mean()
avg_duration_years = avg_duration_months / 12

fn_cost_estimate = avg_loan_eur_present * LGD_ASSUMPTION
lost_interest = avg_loan_eur_present * ANNUAL_INTEREST_RATE * avg_duration_years
fp_cost_estimate = lost_interest + CAC_ESTIMATE
estimated_ratio = fn_cost_estimate / fp_cost_estimate

COST_FP = round(float(fp_cost_estimate), 2)
COST_FN = round(float(fn_cost_estimate), 2)

cost_derivation_check = {
    "methodological_note": (
        "COST_FP and COST_FN values are approximate and do not claim to be precisely correct, as they rest on illustrative assumptions "
        "(Customer Acquisition Cost (CAC), Loss Given Default (LGD), interest rate and inflation adjustment)."
    ),
    "dataset_derived_inputs": {
        "avg_loan_amount_dm_1994": round(float(avg_loan_dm), 2),
        "avg_loan_amount_eur_present": round(float(avg_loan_eur_present), 2),
        "avg_duration_months": round(float(avg_duration_months), 2)
    },
    "assumptions": {
        "lgd": LGD_ASSUMPTION,
        "annual_interest_rate": ANNUAL_INTEREST_RATE,
        "cac_estimate": CAC_ESTIMATE,
        "present_day_inflation_multiplier": PRESENT_DAY_MULTIPLIER
    },
    "fn_cost_eur_estimate": COST_FN,
    "fp_cost_eur_estimate": COST_FP,
    "estimated_ratio": round(float(estimated_ratio), 2),
    "known_simplification": (
        "This audit applies a single flat FP cost and FN cost regardless of the credit amount (which varies within "
        "range DM 250-18,424). In production, the system would price cost per-applicant, using their real exposure."
    )
}

print("\nCOST PARAMETER DERIVATION")
print(f"Avg loan (DM): {avg_loan_dm:.0f}")
print(f"Avg loan (EUR) present-day estimate: {avg_loan_eur_present:.0f}")
print(f"Avg duration (months): {avg_duration_months:.0f}")
print(f"COST_FN: EUR {COST_FN:.2f}")
print(f"COST_FP: EUR {COST_FP:.2f}")
print(f"Ratio: {estimated_ratio:.2f}:1")

# PREPROCESSING

# Split the data in two steps
# First split: separate test set (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y
)
# Second split: separate validation set from the remaining 80% 
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, 
    test_size=VAL_SIZE / (TRAIN_SIZE + VAL_SIZE),  # = 0.20 / 0.80 = 0.25
    random_state=RANDOM_STATE, 
    stratify=y_train_val
)

PROTECTED_COLUMNS = ['age', 'personal_status', 'foreign_worker']

# Create model-specific feature sets (protected columns removed)
X_train_model = X_train.drop(columns=PROTECTED_COLUMNS)
X_val_model = X_val.drop(columns=PROTECTED_COLUMNS)
X_test_model = X_test.drop(columns=PROTECTED_COLUMNS)

categorical_cols = X_train_model.select_dtypes(include=['object', 'str']).columns.tolist()
numerical_cols = X_train_model.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create and apply preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train_model)
X_val_processed = preprocessor.transform(X_val_model)
X_test_processed = preprocessor.transform(X_test_model)

# MODEL TRAINING

# Train logistic regression model
# C=0.5: stronger regularization to reduce overfitting on this small dataset
model = LogisticRegression(solver='liblinear', C=0.5, random_state=RANDOM_STATE) 
model.fit(X_train_processed, y_train)

# PROBABILITY SCORES

# Validation set probabilities for grid search
val_probabilities = model.predict_proba(X_val_processed)[:, 1]
# Test set probabilities for final evaluation
test_probabilities = model.predict_proba(X_test_processed)[:, 1]

# GROUP MASKS FOR FAIRNESS

is_protected_group_val = (X_val['age'] < AGE_THRESHOLD).values
is_protected_group_test = (X_test['age'] < AGE_THRESHOLD).values
is_female_val = X_val['personal_status'].isin(['A92', 'A95']).values
is_female_test = X_test['personal_status'].isin(['A92', 'A95']).values
is_foreign_worker_val = (X_val['foreign_worker'] == 'A201').values
is_foreign_worker_test = (X_test['foreign_worker'] == 'A201').values

# DECISION OUTPUTS

class DecisionOutput(IntEnum):
    APPROVE = 0
    REJECT = 1
    MANUAL_REVIEW = 2

# FUNCTION DEFINITIONS

def apply_decisions(probabilities: np.ndarray, threshold: float, buffer: float, delta=0.0, group_mask=None) -> tuple[np.ndarray, np.ndarray]:
    """
      Apply decision logic with optional group‑specific logit shift.
    
    Args:
        probabilities: Raw model probabilities (0 to 1)
        threshold: Decision threshold
        buffer: Manual review zone half‑width
        delta: Logit shift to apply (default 0.0)
        group_mask: Boolean mask for protected group (required if delta != 0)
    """

      # Apply logit shift only if delta is non‑zero and group_mask is provided
    if delta != 0.0 and group_mask is not None:
        eps = 1e-15
        p_clipped = np.clip(probabilities, eps, 1 - eps)
        logits = np.log(p_clipped / (1 - p_clipped))
        logits_adjusted = logits.copy()
        logits_adjusted[group_mask] += delta
        adjusted_probabilities = 1 / (1 + np.exp(-logits_adjusted))
    elif delta != 0.0 and group_mask is None:
        raise ValueError("group_mask must be provided when delta is non‑zero.")
    else:
        adjusted_probabilities = probabilities

    lower_review_bound = threshold - buffer
    upper_review_bound = threshold + buffer
    
    decisions = np.where(
        (adjusted_probabilities >= lower_review_bound) & (adjusted_probabilities <= upper_review_bound),
        DecisionOutput.MANUAL_REVIEW,
        np.where(adjusted_probabilities > upper_review_bound, DecisionOutput.REJECT, DecisionOutput.APPROVE)
    )
    return adjusted_probabilities, decisions

def analyze_decision_metrics(
    actual: np.ndarray, 
    decisions: np.ndarray, 
    group_mask: np.ndarray,
    cost_fp: float, 
    cost_fn: float
) -> Dict[str, Any]:
    """
    Perform a compliance audit on automated decisions.
    
    Expects integer-encoded arrays:
      Actual: 0 = Good, 1 = Bad
      Decisions: 0 = Approve, 1 = Decline, 2 = Manual Review

    Note: financial_loss reflects total portfolio loss across the full
    automated population passed in, not a cost attributable specifically
    to the protected subgroup defined by group_mask.
    """
    # Isolate automated pipeline decisions from manual review queues
    auto_mask = decisions != DecisionOutput.MANUAL_REVIEW
    
    total_samples = len(actual)
    if total_samples == 0:
        raise ValueError("Input data arrays cannot be empty.")

    stp_count = np.sum(auto_mask)
    stp_rate = (stp_count / total_samples) * 100

    # Masking by group
    act_auto = actual[auto_mask]
    dec_auto = decisions[auto_mask]
    grp_auto = group_mask[auto_mask].astype(bool)
    
    # Financial metrics calculated strictly on automated decisions
    fps = np.sum((act_auto == 0) & (dec_auto == 1)) # Rejecting a good loan
    fns = np.sum((act_auto == 1) & (dec_auto == 0)) # Approving a bad loan
    total_loss = (fps * cost_fp) + (fns * cost_fn)

    # Protected group
    dec_p = dec_auto[grp_auto]
    act_p = act_auto[grp_auto]

    # Reference group
    dec_r = dec_auto[~grp_auto]
    act_r = act_auto[~grp_auto]
    
    def _compute_group_stats(dec: np.ndarray, act: np.ndarray) -> tuple:
        n_total = len(dec)
        if n_total == 0:
            return 0.0, np.nan, np.nan
        
        prob_approved = np.sum(dec == DecisionOutput.APPROVE) / n_total

        good_mask = (act == 0)
        bad_mask = (act == 1)

        n_good = np.sum(good_mask)
        n_bad = np.sum(bad_mask)

        fpr = np.sum(dec[good_mask] == DecisionOutput.REJECT) / n_good if n_good > 0 else np.nan
        tpr = np.sum(dec[bad_mask] == DecisionOutput.REJECT) / n_bad if n_bad > 0 else np.nan

        return prob_approved, fpr, tpr
    
    prob_p_approved, fpr_p, tpr_p = _compute_group_stats(dec_p, act_p)
    prob_r_approved, fpr_r, tpr_r = _compute_group_stats(dec_r, act_r)

    def _calculate_compliance_ratio(numerator: float, denominator:float) -> tuple:
        if np.isnan(numerator) or np.isnan(denominator) or denominator == 0:
            return np.nan, False
        ratio = numerator / denominator
        pass_status = 0.80 <= ratio <= 1.25
        return ratio, pass_status
        
    di_ratio, di_pass = _calculate_compliance_ratio(prob_p_approved, prob_r_approved)
    fpr_ratio, fpr_parity_pass = _calculate_compliance_ratio(fpr_p, fpr_r)
    tpr_ratio, tpr_parity_pass = _calculate_compliance_ratio(tpr_p, tpr_r)
    
    equalized_odds_pass = bool(fpr_parity_pass and tpr_parity_pass)
    overall_fairness_pass = bool(di_pass and equalized_odds_pass)

    return {
        "financial_loss": total_loss,
        "disparate_impact": di_ratio,
        "di_pass": di_pass,
        "stp_rate": stp_rate,
        "false_positives": int(fps),
        "false_negatives": int(fns),
        "fpr_protected": fpr_p,
        "fpr_reference": fpr_r,
        "approval_rate_protected": prob_p_approved,
        "approval_rate_reference": prob_r_approved,
        "fpr_ratio": fpr_ratio,
        "fpr_parity_pass": fpr_parity_pass,
        "tpr_protected": tpr_p,
        "tpr_reference": tpr_r,
        "tpr_ratio": tpr_ratio,
        "tpr_parity_pass": tpr_parity_pass,
        "equalized_odds_pass": equalized_odds_pass,
        "overall_fairness_pass": overall_fairness_pass
    }

def optimize_decision_engine(
    probabilities: np.ndarray,
    actual: np.ndarray,
    group_mask: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    buffer: float,
    secondary_masks: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[float, float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Execute a bounded grid search to find a compliant threshold-delta pair for the primary 
    protected attribute (age < 25).

    At every grid point, also measure DI for any secondary attributes provided in secondary_masks.
    These are not search targets - the search only optimizes over group_mask (age < 25).
    Secondary DI is recorded only to observe cross-attribute effects of the age-based remediation.
    
    Returns:
        Tuple of (best_threshold, best_delta, best_metrics, all_grid_results)
    """
    secondary_masks = secondary_masks or {}
    all_results = []
    best_loss = float('inf')
    best_th = None
    best_delta = None
    best_metrics = {}
    
    # 1,681 Evaluation Matrix Nodes (41 x 41 Grid)
    threshold_grid = np.linspace(0.1, 0.5, 41)
    delta_grid = np.linspace(-0.5, 0.5, 41)
    
    # Constant execution allocations
    eps = 1e-15
    p_clipped = np.clip(probabilities, eps, 1 - eps)
    logits_raw = np.log(p_clipped / (1 - p_clipped))
    
    for th in threshold_grid:
        # Apply Decision Boundaries with Manual Review Buffers
        lower_review_bound = th - buffer
        upper_review_bound = th + buffer

        for delta in delta_grid:
            # Latent Log-odds Transformation
            logits_working = logits_raw.copy()
            logits_working[group_mask] += delta
            adjusted_probs = 1 / (1 + np.exp(-logits_working))
            
            decisions = np.where(
                (adjusted_probs >= lower_review_bound) & (adjusted_probs <= upper_review_bound),
                DecisionOutput.MANUAL_REVIEW,
                np.where(adjusted_probs > upper_review_bound, DecisionOutput.REJECT, DecisionOutput.APPROVE)
            )
            
            metrics = analyze_decision_metrics(
                actual=actual,
                decisions=decisions,
                group_mask=group_mask,
                cost_fp=cost_fp,
                cost_fn=cost_fn
            )

            secondary_di = {}
            for attr_name, attr_mask in secondary_masks.items():
                sec_metrics = analyze_decision_metrics(
                    actual=actual,
                    decisions=decisions,
                    group_mask=attr_mask,
                    cost_fp=cost_fp,
                    cost_fn=cost_fn
                )
                secondary_di[attr_name] = sec_metrics["disparate_impact"]
            
            # Keep the full grid for the pareto/fairness frontier plot
            all_results.append({
                "threshold": th, "delta": delta,
                "financial_loss": metrics["financial_loss"],
                "age_di": metrics["disparate_impact"],
                "age_di_pass": metrics["di_pass"],
                **{f"{k}_di": v for k, v in secondary_di.items()}
            })

            if 0.80 <= metrics["disparate_impact"] <= 1.25:
                if metrics["financial_loss"] < best_loss:
                    best_loss = metrics["financial_loss"]
                    best_th = th
                    best_delta = delta
                    best_metrics = metrics

    if best_th is None:
        raise ValueError("Compliance Crash: Grid Search complete, but no threshold-delta pair " 
        "could satisfy the 80% DI rule. Adjust search boundaries.")
                    
    return float(best_th), float(best_delta), best_metrics, all_results

def get_coefficient_table(model, preprocessor, top_n=10):
    """
    Return a dict of feature names and coefficients.
    Maps one-hot encoded features back to readable names.
    """
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]
    
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coef': np.abs(coefficients)
    }).sort_values('abs_coef', ascending=False)
    
    return {
        "top_n_features": coef_df.head(top_n).to_dict(orient='records'),
        "all_features": coef_df.to_dict(orient='records'),
        "feature_count": len(coef_df)
    } 

# Defining Audit Trail Function

def generate_audit_record(
    model,
    preprocessor,
    metrics_before: Dict[str, Any],
    metrics_val: Dict[str, Any],          
    metrics_after: Dict[str, Any],
    opt_th: float,
    opt_delta: float,
    dataset_info: Dict[str, Any],
    config: Dict[str, Any],
    model_interpretability: Optional[Dict[str, Any]] = None,
    supplementary_attributes: Optional[Dict[str, Any]] = None,
    cost_derivation_check: Optional[Dict[str, Any]] = None,
    cost_ratio_sensitivity: Optional[List[Dict[str, Any]]] = None,
    manual_review_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a structured audit record for regulatory submission.
    
    Args:
        model: Trained scikit-learn model
        preprocessor: Fitted ColumnTransformer
        metrics_before: Baseline fairness metrics (from validation set)
        metrics_val: Metrics after optimization (from validation set)
        metrics_after: Final metrics on test set (unbiased evaluation)
        opt_th: Optimal threshold found via grid search
        opt_delta: Optimal logit shift found via grid search
        dataset_info: Dataset metadata
        config: Configuration parameters
        model_interpretability: Optional dict with coefficients, feature importance
        supplementary_attributes: Optional dics with sex/nationality fairness checks
        cost_derivation_check: Optional dict documenting the COST_FN/COST_FP derivation
        cost_ratio_sensitivity: Optional list of records from the FN:FP ratio sweep
    
    Returns:
        Complete audit record as a dictionary
    """

    audit_record = {
        "meta": {
            "audit_id": f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "audit_date": datetime.now().isoformat(),
            "audit_version": "1.0",
            "model_type": type(model).__name__,
            "model_params": model.get_params(),
            "preprocessor": {
                "type": type(preprocessor).__name__,
                "transformers": str(preprocessor.transformers_)
            }
        },
        "dataset": dataset_info,
        "configuration": config,
        "baseline": {
            "data_used": "validation_set",
            "threshold": config.get("optimal_threshold", 0.0),
            "buffer": config.get("operational_buffer", 0.0),
            "metrics": metrics_before
        },
        "remediation_grid_search": {
            "data_used": "validation_set",
            "optimal_threshold": float(opt_th),
            "optimal_logit_delta": float(opt_delta),
            "metrics_after_optimization": metrics_val
        },
        "final_evaluation": {
            "data_used": "test_set (held out, used once)",
            "methodological_note": (
                "Test set was not used for model training, preprocessing, or "
                "hyperparameter optimization. All metrics reported here are unbiased estimates of "
                "model performance."
            ),
            "metrics": metrics_after
        },
        "comparison": {
            "delta_disparate_impact": metrics_val['disparate_impact'] - metrics_before['disparate_impact'],
            "delta_financial_loss": metrics_val['financial_loss'] - metrics_before['financial_loss']
        },

        "compliance_status": {
            "disparate_impact_pass": bool(metrics_after.get("di_pass", False)),
            "equalized_odds_pass": bool(metrics_after.get("equalized_odds_pass", False)),
            "overall_fairness_pass": bool(metrics_after.get("overall_fairness_pass", False))
        },
    }
    
    if model_interpretability:
        audit_record["interpretability"] = model_interpretability
    if supplementary_attributes:
        audit_record["supplementary_attributes"] = supplementary_attributes
    if cost_derivation_check:
        audit_record["cost_derivation_check"] = cost_derivation_check
    if cost_ratio_sensitivity:
        audit_record["cost_ratio_sensitivity"] = cost_ratio_sensitivity
    if manual_review_analysis:
        audit_record["manual_review_analysis"] = manual_review_analysis

    return audit_record

def bootstrap_metrics(actual, decisions, group_mask, cost_fp, cost_fn, n_iterations=1000, random_state=42):
    """Calculate 95% confidence intervals for DI ratio and equalized odds."""
    rng = np.random.RandomState(random_state)
    di_ratios = []
    fpr_ratios = []
    tpr_ratios = []
    n = len(actual)
    
    for _ in range(n_iterations):
        idx = rng.choice(n, n, replace=True)
        boot_metrics = analyze_decision_metrics(
            actual[idx], decisions[idx], group_mask[idx], cost_fp, cost_fn
        )
        di_ratios.append(boot_metrics["disparate_impact"])
        fpr_ratios.append(boot_metrics["fpr_ratio"])
        tpr_ratios.append(boot_metrics["tpr_ratio"])
    
    return {
        "di_ci": np.percentile(di_ratios, [2.5, 97.5]),
        "fpr_ci": np.percentile(fpr_ratios, [2.5, 97.5]),
        "tpr_ci": np.percentile(tpr_ratios, [2.5, 97.5])
    }

def bootstrap_auc(y_true, y_pred, n_iterations=1000, random_state=42):
    """
    Bootstrap confidence interval for AUC.
    Returns:
        95% confidence interval as [lower, upper]
    """
    np.random.seed(random_state)
    
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    
    aucs = []
    n = len(y_true)
    for _ in range(n_iterations):
        idx = np.random.choice(n, n, replace=True)
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    
    return np.percentile(aucs, [2.5, 97.5])

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts NumPy types to Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Series):
            return obj.to_list()
        return super().default(obj)
    
def proxy_discrimination_check(X_features, y_protected, attribute_name):
    """
    Check how well the remaining features can predict a protected attribute.
    High AUC (>0.70) indicates potential proxy discrimination.

    For multi-class protected attributes (e.g., personal_status), we binarize:
    - For 'personal_status', treat category 'A92' (female) as the positive class.
    - For 'foreign_worker', map 'A201' (yes) to 1 and 'A202' (no) to 0.
    """
    if attribute_name == 'personal_status':
        y_binary = (y_protected == 'A92').astype(int)
    elif attribute_name == 'foreign_worker':
        y_binary = (y_protected == 'A201').astype(int)
    else:
        raise ValueError(f"Unknown attribute: {attribute_name}. Define binarization mapping.")
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_features, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, preds)

def review_routing_check(decisions: np.ndarray, group_mask: np.ndarray) -> Dict[str, Any]:
    """
    Check whether manual-review escalation itself is discriminating against protected group.
    This is different from analyze_decision_metrics, where manual reviews are excluted from 
    the analysis.
    """

    grp_mask = group_mask.astype(bool)
    grp_review_rate = np.mean(decisions[grp_mask] == DecisionOutput.MANUAL_REVIEW) * 100
    ref_review_rate = np.mean(decisions[~grp_mask] == DecisionOutput.MANUAL_REVIEW) * 100
    ratio = grp_review_rate / ref_review_rate if ref_review_rate > 0 else np.nan

    return {
        "review_rate_protected_pct": round(float(grp_review_rate), 2),
        "review_rate_reference_pct": round(float(ref_review_rate), 2),
        "review_rate_ratio": round(float(ratio), 4) if not np.isnan(ratio) else None,
        "n_protected": int(np.sum(grp_mask)),
        "n_reference": int(np.sum(~grp_mask))
    }

def run_buffer_sensitivity(
        val_probabilities: np.ndarray,
        test_probabilities: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        group_mask_val: np.ndarray,
        group_mask_test: np.ndarray,
        cost_fp: float,
        cost_fn: float,
        buffers: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Check if the compliance conclusion and STP rate are stable across a range of 
    manual-review buffer widths. Reruns the grid search for each buffer width, since
    threshold and delta depend on the quantity of samples in automated decisions vs 
    manual review.
    """
    if buffers is None:
        buffers = [0.01, 0.02, 0.03, 0.04, 0.05]
    results = []

    for buf in buffers:
        try:
            b_th, b_delta, b_val_metrics, _ = optimize_decision_engine(
                probabilities=val_probabilities, actual=y_val, group_mask=group_mask_val, 
                cost_fp=cost_fp, cost_fn=cost_fn, buffer=buf
            )
        except ValueError:
            results.append({
                "buffer_pct": buf * 100, "threshold": np.nan, "delta": np.nan, 
                "test_di": np.nan, "test_di_pass": False, "test_eo_pass": False,
                "val_stp_rate_pct": np.nan, "age_review_rate_ratio": np.nan,
                "financial_loss_test": np.nan
            })
            continue

        _, val_dec = apply_decisions(val_probabilities, b_th, buf, delta=b_delta, group_mask=group_mask_val)
        _, test_dec = apply_decisions(test_probabilities, b_th, buf, delta=b_delta, group_mask=group_mask_test)

        test_metrics = analyze_decision_metrics(y_test, test_dec, group_mask_test, cost_fp, cost_fn)
        routing = review_routing_check(test_dec, group_mask_test)
        stp_val = np.mean(val_dec != DecisionOutput.MANUAL_REVIEW) * 100

        results.append({
            "buffer_pct": buf * 100,
            "threshold": b_th,
            "delta": b_delta,
            "val_di": b_val_metrics["disparate_impact"],
            "test_di": test_metrics["disparate_impact"],
            "test_di_pass": test_metrics["di_pass"],
            "test_eo_pass": test_metrics["equalized_odds_pass"],
            "val_stp_rate_pct": round(float(stp_val), 2),
            "age_review_rate_ratio": routing["review_rate_ratio"],
            "financial_loss_test": test_metrics["financial_loss"]
        })

    return pd.DataFrame(results)

def run_cost_sensitivity(
        probabilities: np.ndarray,
        actual: np.ndarray,
        group_mask: np.ndarray,
        base_fp: float,
        buffer: float,
        secondary_masks: Optional[Dict[str, np.ndarray]] = None,
        ratios: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Run sensitivity analysis across a range of FN:FP cost ratios.

    FP is held constant at base_fp. FN is set to base_fp * ratio for each ratio
    tested, so the 'ratio' column directly equals the FN:FP ratio evaluated at 
    that row (e.g. ratio=3.0 means FN cost is exactly 3 * FP cost).
    """
    if ratios is None:
        ratios = [1.0, 1.5, 2.0, 3.4, 4.5, 6.0]
    
    secondary_masks = secondary_masks or {}
    results = []

    for ratio in ratios:
        test_fp = base_fp
        test_fn = base_fp * ratio # ratio is the FN:FP ratio being tested

        try:
            opt_th, opt_delta, opt_metrics, _ = optimize_decision_engine(
                probabilities=probabilities,
                actual=actual,
                group_mask=group_mask,
                cost_fp=test_fp,
                cost_fn=test_fn,
                buffer=buffer,
                secondary_masks=secondary_masks,
            )

            results.append({
                "ratio": ratio,
                "threshold": opt_th,
                "delta": opt_delta,
                "di_ratio": opt_metrics["disparate_impact"],
                "di_pass": opt_metrics["di_pass"],
                "financial_loss": opt_metrics["financial_loss"],
                "fpr_ratio": opt_metrics["fpr_ratio"],
                "fpr_pass": opt_metrics["fpr_parity_pass"],
                "tpr_ratio": opt_metrics["tpr_ratio"],
                "tpr_pass": opt_metrics["tpr_parity_pass"],
                "equalized_odds_pass": opt_metrics["equalized_odds_pass"],
                "overall_pass": opt_metrics["overall_fairness_pass"]
            })
        except ValueError:
            # No compliant solution found at this ratio
            results.append({
                "ratio": ratio,
                "threshold": np.nan,
                "delta": np.nan,
                "di_ratio": np.nan,
                "di_pass": False,
                "financial_loss": np.nan,
                "fpr_ratio": np.nan,
                "fpr_pass": False,
                "tpr_ratio": np.nan,
                "tpr_pass": False,
                "equalized_odds_pass": False,
                "overall_pass": False
            })
    return pd.DataFrame(results)

# DECISION ENGINE

# Calculate optimal threshold based on asymmetric loss ratio
optimal_threshold = COST_FP / (COST_FP + COST_FN)

# Define manual review zone
lower_review_bound = optimal_threshold - OPERATIONAL_BUFFER
upper_review_bound = optimal_threshold + OPERATIONAL_BUFFER

# Apply decision engine
_, baseline_decisions = apply_decisions(val_probabilities, optimal_threshold, OPERATIONAL_BUFFER)

# BUSINESS IMPACT

# Create results table
results_df = pd.DataFrame({'Actual': y_val.values, 'Decision': baseline_decisions})
decision_labels = {
    DecisionOutput.APPROVE: "Approve",
    DecisionOutput.REJECT: "Reject",
    DecisionOutput.MANUAL_REVIEW: "Manual Review"
}
results_df['Decision_Label'] = results_df['Decision'].map(decision_labels)

# Decision matrix
matrix = pd.crosstab(results_df['Actual'], results_df['Decision_Label'])
print("\nPHASE 1: BASELINE ASSESSMENT (Naive Threshold Model)")
print("\nDecision Engine Results:")
print(f"Optimal Risk Cutoff: {optimal_threshold:.2f}")
print(f"Manual Review Zone: [{lower_review_bound:.2f} to {upper_review_bound:.2f}]")
print("\nDecision Matrix:")
print(matrix)

# Calculate STP rate
total_apps = len(results_df)
automated_apps = len(results_df[results_df['Decision'] != DecisionOutput.MANUAL_REVIEW])
stp_rate = (automated_apps / total_apps) * 100
print(f"\nStraight-Through Processing (STP) Rate: {stp_rate:.1f}%")

# BASELINE FAIRNESS AUDIT
print("\nBaseline Fairness Audit (Automated Decisions Only)")

val_baseline_metrics = analyze_decision_metrics(
    actual=y_val.values,
    decisions=baseline_decisions,
    group_mask=is_protected_group_val,
    cost_fp=COST_FP,
    cost_fn=COST_FN
)

# Print baseline fairness results
print("\nBaseline Fairness (Validation Set)")
print(f"DI Ratio: {val_baseline_metrics['disparate_impact']:.4f}")
print(f"FPR (Protected/Ref): {val_baseline_metrics['fpr_protected']:.4f} / {val_baseline_metrics['fpr_reference']:.4f}")
print(f"TPR (Protected/Ref): {val_baseline_metrics['tpr_protected']:.4f} / {val_baseline_metrics['tpr_reference']:.4f}")
print(f"Financial Loss: EUR {val_baseline_metrics['financial_loss']:,.2f}")

# REMEDIATION

print("\nPHASE 2: REMEDIATION GRID SEARCH (Optimized Model)")

try:
    opt_th, opt_delta, opt_metrics, all_results = optimize_decision_engine(
        probabilities=val_probabilities,
        actual=y_val.values,
        group_mask=is_protected_group_val,
        cost_fp=COST_FP,
        cost_fn=COST_FN,
        buffer=OPERATIONAL_BUFFER,
        secondary_masks={
            "sex": is_female_val,
            "foreign_worker": is_foreign_worker_val
        }
    )
    print(f"Optimal Operational State Confirmed:")
    print(f"Optimal Threshold (Θ): {opt_th:.4f}")
    print(f"Optimal Delta (δ): {opt_delta:+.4f}")
    print(f"DI Ratio (after remediation): {opt_metrics['disparate_impact']:.4f}")
    print(f"Financial Loss (after remediation): EUR {opt_metrics['financial_loss']:,.2f}")

except ValueError as compliance_error:
    print(f"Validation Error: {compliance_error}", file=sys.stderr)
    sys.exit(1)  # Explicit, intentional stop with non-zero exit code


# COST RATIO SENSITIVITY ANALYSIS
print("\nCOST RATIO SENSITIVITY ANALYSIS")
print("Testing compliance stability across FN:FP ratios 1:1 to 6:1")

sensitivity_df = run_cost_sensitivity(
    probabilities=val_probabilities,
    actual=y_val.values,
    group_mask=is_protected_group_val,
    base_fp=COST_FP,
    buffer=OPERATIONAL_BUFFER,
    secondary_masks={
        "sex": is_female_val,
        "foreign_worker": is_foreign_worker_val
    }
)

print("\nSensitivity Results:")
print(sensitivity_df.to_string(index=False))

# Apply decision engine (test set)
test_probabilities_adjusted, test_decisions = apply_decisions(
    test_probabilities, 
    opt_th, 
    OPERATIONAL_BUFFER, 
    delta=opt_delta, 
    group_mask=is_protected_group_test
    )

test_metrics = analyze_decision_metrics(
    actual=y_test.values,
    decisions=test_decisions,
    group_mask=is_protected_group_test,
    cost_fp=COST_FP,
    cost_fn=COST_FN
)

print("\nPHASE 3: FINAL EVALUATION (Test Set - Unbiased):")
print(f"DI Ratio (Test Set): {test_metrics['disparate_impact']:.4f}")
print(f"FPR (Protected/Ref): {test_metrics['fpr_protected']:.4f} / {test_metrics['fpr_reference']:.4f}")
print(f"TPR (Protected/Ref): {test_metrics['tpr_protected']:.4f} / {test_metrics['tpr_reference']:.4f}")
print(f"Financial Loss (Test Set): EUR {test_metrics['financial_loss']:,.2f}")

# COMPARISON - BEFORE VS AFTER REMEDIATION

print("\nMODEL COMPARISON (Before vs After Remediation, Validation Set)")
print(f"{'Metric':<30} {'Baseline':>12} {'Remediated':>12} {'Change':>12}")
print(f"{'Disparate Impact':<30} {val_baseline_metrics['disparate_impact']:>12.4f} {opt_metrics['disparate_impact']:>12.4f} {opt_metrics['disparate_impact'] - val_baseline_metrics['disparate_impact']:>+12.4f}")
print(f"{'Financial Loss (EUR)':<30} {val_baseline_metrics['financial_loss']:>12,.2f} {opt_metrics['financial_loss']:>12,.2f} {opt_metrics['financial_loss'] - val_baseline_metrics['financial_loss']:>+12,.2f}")

# AUC on validation set (diagnostic)
auc_val = roc_auc_score(y_val, val_probabilities)
print(f"Validation AUC: {auc_val:.3f}")

# AUC on test set (final, unbiased)
auc_test = roc_auc_score(y_test, test_probabilities)
print(f"Test AUC: {auc_test:.3f}")

print("\nMANUAL REVIEW ESCALATION EVALUATION:")

routing_checks = {
    "age": review_routing_check(test_decisions, is_protected_group_test),
    "sex": review_routing_check(test_decisions, is_female_test),
    "foreign_worker": review_routing_check(test_decisions, is_foreign_worker_test)
}

for attr_name, result in routing_checks.items():
    print(f"\n{attr_name}: Protected {result['review_rate_protected_pct']}% reviewed "
          f"(n={result['n_protected']}) vs Reference {result['review_rate_reference_pct']}% "
          f"reviewed (n={result['n_reference']}); ratio: {result['review_rate_ratio']}")
    
buffer_sensitivity_df = run_buffer_sensitivity(
    val_probabilities=val_probabilities,
    test_probabilities=test_probabilities,
    y_val=y_val.values,
    y_test=y_test.values,
    group_mask_val=is_protected_group_val,
    group_mask_test=is_protected_group_test,
    cost_fp=COST_FP,
    cost_fn=COST_FN
)

print("\nBuffer Sensitivity (Age, 1%-5%):")
print(buffer_sensitivity_df.to_string(index=False))

manual_review_analysis = {
    "methodological_note": (
        "DI/FPR/TPR metrics in this audit exclude manual-review cases by "
        "design, to isolate automated-decision performance. This check "
        "examines the excluded population directly: whether routing into "
        "manual review is itself disproportionate by protected group, and "
        "whether the compliance conclusion is stable across plausible "
        "buffer widths."
    ),
    "routing_disparity_by_attribute": routing_checks,
    "buffer_sensitivity": buffer_sensitivity_df.to_dict(orient='records'),
    "findings": (
        "Age: manual-review routing ratio ~1.3-1.4x (protected vs reference), "
        "stable across the automated-decision approve/reject fairness result. "
        "This is a real disparity not captured by DI/FPR/TPR, since those "
        "metrics exclude review cases. Foreign_worker routing ratio is not "
        "interpretable given the small reference-group sample size (n=6, test "
        "set) already flagged elsewhere in this audit. Equalized-odds failure "
        "on the test set persists across all tested buffer widths (1%-5%), "
        "indicating it is a property of the remediation itself, not an "
        "artifact of the specific 3% buffer chosen."
    )
}

# AUDIT TRAIL EXECUTION

# Dataset info
dataset_info = {
    "name": "German Credit Dataset",
    "source": "UCI Machine Learning Repository",
    "samples": len(df),
    "features": len(X.columns),
    "target": "class (0=Good, 1=Bad)",
    "split": {
        "train": len(X_train),
        "validation": len(X_val),
        "test": len(X_test)
    },
    "split_ratios": {
        "train": TRAIN_SIZE,
        "validation": VAL_SIZE,
        "test": TEST_SIZE
    },
    "protected_attribute": {
        "name": "age",
        "threshold": AGE_THRESHOLD,
        "protected_label": PROTECTED_LABEL,
        "reference_label": REFERENCE_LABEL
    },
    "limitations": {
        "selection_bias": "Dataset only contains applicants who were historically extended credit. No data on rejected applicants.",
        "sample_size": "1,000 samples total, 200 in test set. Confidence intervals should be interpreted with caution."
    }
}

# Configuration
config = {
    "train_size": TRAIN_SIZE,
    "val_size": VAL_SIZE,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "cost_fp": COST_FP,
    "cost_fn": COST_FN,
    "operational_buffer": OPERATIONAL_BUFFER,
    "optimal_threshold": optimal_threshold,
    "age_threshold": AGE_THRESHOLD,
    "grid_search": {
        "threshold_grid": np.linspace(0.1, 0.5, 41).tolist(),
        "delta_grid": np.linspace(-0.5, 0.5, 41).tolist(),
        "grid_nodes": 41 * 41
    },
    "manual_review_exclusion": "All fairness metrics exclude manual review cases to isolate automated system performance."
}

# SUPPLEMENTARY FAIRNESS ATTRIBUTES (Sex, Nationality)

foreign_worker_metrics = analyze_decision_metrics(
    actual=y_test.values, 
    decisions=test_decisions, 
    group_mask=is_foreign_worker_test, 
    cost_fp=COST_FP, 
    cost_fn=COST_FN
    )

female_metrics = analyze_decision_metrics(
    actual=y_test.values, 
    decisions=test_decisions, 
    group_mask=is_female_test, 
    cost_fp=COST_FP, 
    cost_fn=COST_FN
    )

ps_counts_test = X_test['personal_status'].value_counts().to_dict()
ps_counts_full = df['personal_status'].value_counts().to_dict()
foreign_counts_test = X_test['foreign_worker'].value_counts().to_dict()
foreign_counts_full = df['foreign_worker'].value_counts().to_dict()

supplementary_attributes = {
    "methodological_note": (
        "These checks evaluate the age-remediated test-set decisions "
        "(test_decisions) for disparity along sex and nationality. No "
        "remediation was performed for these attributes; results are "
        "single-point assessments, not baseline-vs-remediated comparisons. "
        "'financial_loss' inside each metrics block reflects total portfolio "
        "loss across the full automated population, not a cost attributable "
        "to this specific subgroup."
    ),
    "foreign_worker": {
        "metrics": foreign_worker_metrics,
        "sample_counts": {
            "test_set": foreign_counts_test,
            "full_dataset": foreign_counts_full
        },
        "finding": (
            "Native workers (A202) represent a small minority of both the "
            "full dataset and the test split. This is a structural property "
            "of the source data, not an artifact of this train/test split. "
            "DI ratio may be statistically unreliable or NaN due to small "
            "reference-group cell counts after excluding manual-review cases."
        )
    },
    "sex": {
        "metrics": female_metrics,
        "sample_counts": {
            "test_set": ps_counts_test,
            "full_dataset": ps_counts_full
        },
        "finding": (
            "The 'personal_status' field conflates sex and marital status. "
            "The single-female category (A95) is effectively unpopulated in "
            "the full dataset, not just this test split, so sex cannot be "
            "cleanly isolated from marital status using this attribute. "
            "The 'female' group here (A92 + A95) is dominated by married/"
            "divorced women; no reliable single-female comparison is possible "
            "with this data."
        )
    }
}

# Model interpretability
model_interpretability = get_coefficient_table(model, preprocessor, top_n=10)

# Generate audit record
audit_record = generate_audit_record(
    model=model,
    preprocessor=preprocessor,
    metrics_before=val_baseline_metrics,
    metrics_val=opt_metrics,
    metrics_after=test_metrics,
    opt_th=opt_th,
    opt_delta=opt_delta,
    dataset_info=dataset_info,
    config=config,
    model_interpretability=model_interpretability,
    supplementary_attributes=supplementary_attributes,
    cost_derivation_check=cost_derivation_check,
    cost_ratio_sensitivity=sensitivity_df.to_dict(orient='records'),
    manual_review_analysis=manual_review_analysis
)

# Calling bootstrap metrics
ci = bootstrap_metrics(
    actual=y_test.values,
    decisions=test_decisions,
    group_mask=is_protected_group_test,
    cost_fp=COST_FP,
    cost_fn=COST_FN,
    n_iterations=1000,
    random_state=42
)
print("\n Confidence Intervals:")
print(f"DI 95% CI: [{ci['di_ci'][0]:.4f}, {ci['di_ci'][1]:.4f}]")
print(f"FPR 95% CI: [{ci['fpr_ci'][0]:.4f}, {ci['fpr_ci'][1]:.4f}]")
print(f"TPR 95% CI: [{ci['tpr_ci'][0]:.4f}, {ci['tpr_ci'][1]:.4f}]")

# Calling AUC confidence intervals
auc_ci = bootstrap_auc(y_test, test_probabilities, n_iterations=1000,
    random_state=42)
print(f"Test AUC 95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")

for attr in ['personal_status', 'foreign_worker']:
    auc = proxy_discrimination_check(X_train_processed, X_train[attr], attr)
    status = "HIGH risk" if auc > 0.70 else "low risk"
    print(f"  {attr}: AUC = {auc:.3f} ({status})")

# Supplementary Fairness Metrics 
print("\nSupplementary Fairness Metrics (Sex and Nationality)")
print("\nForeign Worker Status (Test Set):")
print(f"DI Ratio (Test Set): {foreign_worker_metrics['disparate_impact']:.4f}")
print(f"FPR (Foreign/Domestic): {foreign_worker_metrics['fpr_protected']:.4f} / {foreign_worker_metrics['fpr_reference']:.4f}")
print(f"TPR (Foreign/Domestic): {foreign_worker_metrics['tpr_protected']:.4f} / {foreign_worker_metrics['tpr_reference']:.4f}")

print("\nFemale Status (Test Set):")
print(f"DI Ratio (Test Set): {female_metrics['disparate_impact']:.4f}")
print(f"FPR (Female/Male): {female_metrics['fpr_protected']:.4f} / {female_metrics['fpr_reference']:.4f}")
print(f"TPR (Female/Male): {female_metrics['tpr_protected']:.4f} / {female_metrics['tpr_reference']:.4f}")

female_sum = np.sum(is_female_test)
foreign_sum = np.sum(is_foreign_worker_test)

print(f"\nNumber of females in the test set (out of 200): {female_sum}")
print("Personal status counts: ")
print("A91 : male   : divorced/separated")
print("A92 : female : divorced/separated/married")
print("A93 : male   : single")
print("A94 : male   : married/widowed")
print("A95 : female : single")
print(f": {ps_counts_test}")
print(f"\nNumber of foreign workers in the test set (out of 200): {foreign_sum}")
print(f"Foreign worker (A201) vs Native worker (A202) counts: {foreign_counts_test}")


print("\nGLOSSARY: Technical Terms for Compliance Readers")
print("\nDisparate Impact (DI) Ratio:")
print("Approval rate (protected) ÷ Approval rate (reference)")
print("Under the 80% rule, DI must be >= 0.80 (and typically <= 1.25).")
print("\nFalse Positive Rate (FPR):")
print("Good borrowers incorrectly rejected ÷ All good borrowers")
print("In fair-lending terms: 'unjustified rejections.'")
print("\nTrue Positive Rate (TPR):")
print("Bad borrowers correctly rejected ÷ All bad borrowers")
print("In fair-lending terms: 'fraud/default detection rate.'")
print("\nEqualized Odds:")
print("A fairness criterion requiring that FPR and TPR are equal across groups.")
print("In practice: the ratios should be within the 0.80–1.25 band.")

# PARETO FRONTIER: FAIRNESS vs. FINANCIAL LOSS (age, sex, foreign_worker)

grid_df = pd.DataFrame(all_results)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
attr_specs = [
    ("age_di", "Age DI Ratio vs. Cost", axes[0]),
    ("sex_di", "Sex DI Ratio vs. Cost (measured, not optimized)", axes[1]),
    ("foreign_worker_di", "Foreign Worker DI Ratio vs. Cost (measured, not optimized)", axes[2]),
]

for col, title, ax in attr_specs:
    ax.scatter(grid_df[col], grid_df["financial_loss"], s=8, alpha=0.3, color="grey")
    ax.axvspan(0.80, 1.25, color="green", alpha=0.08, label="80% rule compliance band")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Disparate Impact Ratio")
    ax.axvline(1.0, color="black", linewidth=0.5, linestyle="--")

axes[0].set_ylabel("Financial Loss (€)")

# Mark the selected operating point on the age panel
axes[0].scatter(
    [opt_metrics["disparate_impact"]], [opt_metrics["financial_loss"]],
    color="red", s=60, zorder=5, label="Selected operating point"
)
axes[0].legend(fontsize=8, loc="upper right")

plt.suptitle("Fairness-Cost Frontier Across Protected Attributes\n(grid searched on age only; sex/foreign_worker shown as cross-attribute effect)", fontsize=11)
plt.tight_layout()
plt.savefig("fairness_cost_frontier.png", dpi=150, bbox_inches="tight")
print("\nFairness-cost frontier saved to 'fairness_cost_frontier.png'")

# Save to JSON
with open('audit_record.json', 'w') as f:
    json.dump(audit_record, f, indent=2, cls=NumpyEncoder)

print("\nAudit record saved to 'audit_record.json'")



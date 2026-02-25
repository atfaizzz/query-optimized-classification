#  Innov8 4.0 — Talent Fraud Detection Challenge  
### Sponsored by [Eightfold AI](https://eightfold.ai)

---

#  Our Approach: Budget-Optimized Active Learning System

We implemented a **two-stage active learning framework** designed to maximize fraud detection performance under a strict oracle query budget (100 labels).

Instead of random sampling, our solution strategically allocates the budget across:

1. **Heuristic-driven fraud cluster discovery**
2. **Model-guided uncertainty refinement**

This allows us to extract significantly more fraud signals per query compared to naive sampling.

---

# 🏗️ Strategy Overview

## 🔹 Stage 1 — Targeted Exploration (~70 queries)

We engineered composite heuristic risk scores for each of the four fraud types:

| Fraud Type | Signals Used |
|------------|--------------|
| Credential Fraud | `institution_risk_score`, `gpa_anomaly_score`, `company_risk_score`, `tenure_gap_months` |
| Application Bombing | `applications_7d`, `applications_30d`, `app_to_avg_ratio`, `time_since_last_app_hrs` |
| Account Takeover | `is_new_device`, `failed_logins_24h`, `login_velocity_24h`, `ip_risk_score` |
| Ghost Profile | `copy_paste_ratio`, `profile_age_days`, `skills_to_exp_ratio`, `email_risk_score` |

### What we do:

- Normalize each feature to [0, 1]
- Construct fraud-type-specific composite scores
- Query top-ranked candidates per fraud cluster
- Ensure coverage across all fraud patterns

This dramatically increases the density of fraud examples in the labeled set compared to random sampling (~8%).

---

## 🔹 Stage 2 — Model-Guided Uncertainty Sampling (~30 queries)

After building an initial model, we:

1. Train a balanced ensemble classifier.
2. Predict probabilities on all unlabeled samples.
3. Identify samples closest to the decision boundary (p ≈ 0.5).
4. Combine uncertainty with heuristic risk to prioritize suspicious-but-ambiguous cases.
5. Query those samples.
6. Retrain the model with expanded labeled data.

This step refines the classification boundary and improves recall without sacrificing precision.

---

#  Final Model: Balanced Ensemble

We use a weighted ensemble of:

- **Gradient Boosting (GBM)**
- **Random Forest**
- **Logistic Regression**

### Key design choices:

- `class_weight="balanced"` to handle ~8% fraud imbalance
- Sample weighting via `compute_sample_weight`
- Probability averaging (0.45 GBM / 0.35 RF / 0.20 LR)
- Oracle labels directly injected for queried rows (perfect ground truth)

---

#  Expected Performance

| Method | Expected F1 (Fraud Class) |
|--------|---------------------------|
| Random Sampling Baseline | ~0.30 – 0.45 |
| Our Two-Stage Active Learning | **0.65 – 0.80** |

Performance may vary slightly due to dataset randomness.

---

#  Why This Works

- Fraud clusters are structured — not random.
- Early-stage cluster discovery increases labeled fraud density.
- Balanced ensemble prevents majority-class bias.
- Uncertainty sampling focuses queries on high-information regions.
- Budget is allocated strategically (70/30 split).
- Fewer wasted queries → higher effective learning.

---

#  Implementation Details

- Noise features (`feature_noise_*`) are removed.
- Missing values are median-imputed.
- All heuristics are normalized.
- Safe fallback: If early queries return only one class, composite risk scores are used for prediction.
- Fully compliant with evaluation restrictions (no forbidden imports, no file I/O, no networking).

---

#  Local Testing

```bash
python framework.py --agent agent.py

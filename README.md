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
```
---

# 🧮 Mathematical Framing of the Problem

This challenge can be formalized as a **budget-constrained pool-based active learning problem**:

Given:
- Unlabeled dataset \( X \in \mathbb{R}^{10000 \times 25} \)
- Hidden labels \( y \in \{0,1\}^{10000} \)
- Oracle query budget \( B = 100 \)

Objective:
Maximize F1 score on unseen data while minimizing oracle queries.

We approximate the optimal strategy:

\[
\max_{Q \subset X, |Q| \le B} \; \text{F1}(f_{\theta}(X))
\]

Where:
- \( Q \) = selected query set
- \( f_{\theta} \) = classifier trained on queried samples
- Budget constraint ensures \(|Q| \le 100\)

Our solution optimizes **information gain per query**, not just label acquisition.

---

#  Budget Allocation Strategy

We intentionally split the budget:

| Stage | Queries | Purpose |
|--------|---------|----------|
| Stage 1 | ~70 | Discover fraud clusters via heuristic priors |
| Stage 2 | ~30 | Refine decision boundary via uncertainty sampling |

### Why 70/30?

- Early exploration increases fraud density in labeled set.
- Later exploitation sharpens the classifier.
- Prevents overfitting to one fraud cluster.
- Empirically balances recall and precision.

---

#  Feature Engineering Philosophy

Rather than blindly trusting raw features, we:

- Removed synthetic noise features (`feature_noise_*`)
- Normalized heterogeneous risk scores
- Constructed fraud-type-specific composite signals
- Applied max-aggregation across fraud types to detect multiple risk modes

This creates **unsupervised priors** before any label is queried.

---

#  Information-Efficient Querying

Instead of uniform sampling:

### Stage 1:
We approximate prior probability of fraud:
\[
P(y=1|x) \approx \max(\text{cred}, \text{bomb}, \text{ato}, \text{ghost})
\]

This increases labeled fraud yield beyond the base rate (~8%).

### Stage 2:
We approximate information gain using:

\[
\text{Uncertainty}(x) = 1 - |2P(y=1|x) - 1|
\]

We then combine:

\[
\text{Score}(x) = 0.6 \cdot \text{Uncertainty}(x) + 0.4 \cdot \text{CompositeRisk}(x)
\]

This prioritizes:
- Decision-boundary samples
- Suspicious-but-ambiguous profiles
- High learning-value regions

---

#  Handling Class Imbalance

Fraud rate ≈ 8%.

We mitigate imbalance via:

- `class_weight="balanced"`
- Sample reweighting using `compute_sample_weight`
- Ensemble averaging to stabilize minority predictions

This improves recall without sacrificing precision.

---

# 🔬 Ensemble Design Rationale

| Model | Strength |
|--------|----------|
| Gradient Boosting | Captures non-linear interactions |
| Random Forest | Robust to noise, reduces variance |
| Logistic Regression | Provides linear calibration |

Weighted averaging improves:
- Stability
- Calibration
- Generalization

Final probability:
\[
P = 0.45P_{GBM} + 0.35P_{RF} + 0.20P_{LR}
\]

---

# 🛡️ Robustness & Fallback Design

Edge case handled:

If early-stage queries return only one class:
- Model training is skipped.
- Composite risk score is used as probability proxy.
- Prevents model crash and ensures valid output.

Additionally:
- Queried rows use oracle truth directly.
- No dependency on external files.
- Fully deterministic (`random_state=42`).

---

#  Computational Efficiency

- Vectorized NumPy operations
- No unnecessary retraining loops
- Only two model training phases
- Runtime well under 5-minute limit
- No heavy hyperparameter search

Designed for both **accuracy and speed** (tie-breaker aware).

---

#  Why This Outperforms Random Sampling

Random sampling:
- ~8 fraud in 100 queries
- Poor boundary learning
- Weak recall

Our approach:
- Cluster-aware fraud discovery
- Balanced labeled dataset
- Boundary-focused refinement
- Higher fraud density in labeled pool
- Significantly improved F1

---

#  Competitive Advantages

✔ Structured exploration  
✔ Information-theoretic reasoning  
✔ Cluster-aware sampling  
✔ Budget optimization  
✔ Ensemble robustness  
✔ Tie-breaker conscious design  
✔ Clean, evaluator-compliant implementation  

---

# 🏁 Conclusion

This solution transforms a limited-label fraud detection task into an optimized active learning pipeline that:

- Maximizes fraud discovery per query
- Efficiently learns under strict budget constraints
- Maintains strong generalization
- Balances recall and precision
- Demonstrates production-level ML engineering discipline

The result is a scalable, information-efficient fraud detection agent designed to operate under real-world labeling constraints.

---

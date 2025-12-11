# Week4-credit-risk-model

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord (particularly Pillars 1 and 3) fundamentally shapes credit risk modeling requirements in regulated banking:

- Under the **Internal Ratings-Based (IRB) approach**, banks may use their own Probability of Default (PD) models, but only if the model is rigorously validated, transparently documented, independently validated, and demonstrably conservative.
- **Pillar 3** mandates public and regulatory disclosure of risk measurement methodologies, including how PD is estimated.
- Regulators require **use test** evidence — the model must be actually used in credit decisions and capital allocation, which demands clear interpretability.

Consequently, even when using alternative behavioral data (as in this project), Bati Bank must prioritize:

- Full audit trail of data sources and transformations
- Interpretable features and model logic
- Comprehensive model documentation and governance
- Ability to explain individual decisions (critical for customer rejection fairness under responsible lending rules)

Non-compliance risks regulatory rejection of the model and higher capital requirements under the standardized approach.

### 2. Why is creating a proxy variable necessary, and what are the potential business risks?

The Xente dataset contains rich transactional behavior but **no historical loan repayment outcome or explicit default flag**. Traditional credit scoring relies on observed good/bad performance (e.g., 90+ days past due within 12 months). In the absence of this, we must construct a behavioral proxy for credit risk.

We define **"disengaged/low-engagement" customers** — identified through RFM clustering — as a proxy for high-risk (potential defaulters). This assumes that customers who stop transacting, transact infrequently, or spend very little are more likely to default if granted credit.

**Key Business & Model Risks**:
| Risk | Description | Impact |
|------|-----------|--------|
| Proxy misalignment | Disengaged ≠ Defaulter (e.g., wealthy customer who rarely uses wallet) | High false positives → lost revenue |
| Circular logic | Using spending behavior to predict repayment of spending-based credit | May overfit to current product usage |
| Regulatory challenge | Supervisors may reject purely behavioral PD models without back-testing against actual defaults | Model disapproval, capital penalty |
| Performance decay | Proxy validity erodes if customer behavior shifts post-credit granting | Increased real defaults over time |

Mitigation strategy: Treat this as a **Version 0.1 challenger model**, monitor actual default rates post-launch, and plan rapid recalibration when true outcome data becomes available.

### 3. Key trade-offs: Simple interpretable vs complex high-performance models in regulated finance

| Criterion                      | Logistic Regression + WoE/IV                                | Gradient Boosting (XGBoost/LightGBM/CatBoost)                    |
| ------------------------------ | ----------------------------------------------------------- | ---------------------------------------------------------------- |
| Interpretability               | Excellent – monotonic transformations, direct odds impact   | Low – complex non-linear interactions                            |
| Explainability to customers    | Easy ("Your low transaction frequency increased your risk") | Difficult without SHAP/LIME                                      |
| Regulatory acceptance          | High – industry standard for scorecards                     | Conditional – accepted only with strong explainability framework |
| Performance (ROC-AUC)          | Usually 0.70–0.80 with good features                        | Often 0.80–0.90+                                                 |
| Stability & robustness         | Very high                                                   | Can overfit; sensitive to data drift                             |
| Rejection reasoning            | Straightforward and consistent                              | Complex; requires additional tooling                             |
| Development & maintenance cost | Low                                                         | Higher (tuning, monitoring, explanation layer)                   |
| Recommended for Bati Bank v1   | Yes – as production baseline + scorecard                    | Yes – as challenger or ensemble component with SHAP              |

**Conclusion for this project**:  
We will develop \*\*both:

- A **Logistic Regression + WoE** model → converted into a transparent 300–850 credit scorecard (production baseline, fully compliant)
- A **Gradient Boosting** model → used as high-performance challenger and for ensemble blending (with SHAP explanations logged)

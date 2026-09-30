# Chapter 4 — Results: findings and draft text

Running record. Numbers are final unless marked otherwise. Draft paragraphs are
written to be dropped into the chapter and edited into your own voice.

**Test set:** 874 loans, 190 defaults (21.74%). Used once, after all modelling
decisions were fixed.

---

## 4.1 Discrimination

**Table 4.1 — Discrimination on the test set**

| Model | ROC-AUC | PR-AUC | KS |
|---|---|---|---|
| XGBoost | 0.7254 | 0.4709 | 0.3749 |
| CatBoost | 0.7201 | 0.4571 | 0.3667 |
| Random Forest | 0.7157 | 0.4450 | 0.3661 |
| LightGBM | 0.7142 | 0.4358 | 0.3585 |
| Logistic Regression | 0.7075 | 0.4345 | 0.3664 |

PR-AUC baseline (random ranking) = 0.2174. Spread in ROC-AUC = 0.0179.

**Findings**

- Every model at least doubles the PR-AUC baseline (1.99x to 2.17x).
- KS of 0.359–0.375 is within the range generally considered workable in credit
  scoring practice.
- The ranking differs from the validation ranking, where CatBoost and Logistic
  Regression led. With a spread of 0.018 this reordering is not meaningful.

**Draft text**

> All five models achieved ROC-AUC between 0.708 and 0.725 on the test set, with
> PR-AUC between 0.435 and 0.471 against a baseline of 0.217 for random ranking.
> Every model therefore identified defaulting borrowers at approximately twice
> the rate expected by chance. The Kolmogorov-Smirnov statistic ranged from 0.359
> to 0.375. XGBoost achieved the highest value on all three measures, though the
> spread across models was only 0.018 in ROC-AUC, and the ordering differed from
> that observed on the validation set. Section 4.5 tests whether these
> differences are statistically meaningful.

---

## 4.2 Calibration

**Table 4.2 — Calibration on the test set**

| Model | Brier | Log loss | Mean predicted p | Calibration gap |
|---|---|---|---|---|
| XGBoost | 0.14916 | 0.46770 | 0.2158 | −0.0016 |
| CatBoost | 0.14939 | 0.46889 | 0.2157 | −0.0017 |
| Random Forest | 0.14975 | 0.46926 | 0.2184 | +0.0010 |
| LightGBM | 0.15070 | 0.47014 | 0.2195 | +0.0021 |
| Logistic Regression | 0.15162 | 0.47343 | 0.2130 | −0.0044 |

Baseline (predict the training base rate for every loan): Brier 0.17013,
log loss 0.52359. Best model improves Brier by 12.3%, log loss by 10.7%.

**Findings**

- Mean predicted probability is within 0.0044 of the observed default rate for
  every model, on data the calibrator never saw. The calibration generalised.
- Reliability curves sit slightly above the diagonal between 0.15 and 0.35,
  meaning risk is marginally understated in the band containing p*.
- Predicted probabilities span roughly 0.09 to 0.55. No borrower is assigned a
  risk above 55%, which is why a threshold of 0.5 rejects almost nobody.
- Platt scaling produced near-zero change in Brier score on validation
  (e.g. 0.14927 to 0.14915 for LR), indicating the models were already close to
  calibrated before adjustment. Isotonic regression appeared to improve Brier
  substantially (to 0.1440) but this was in-sample fitting: it raised CatBoost's
  validation AUC from 0.7239 to 0.7372, which a monotonic transform cannot
  legitimately do.

**Draft text**

> Calibration transferred well to the test set: mean predicted probability lay
> within 0.005 of the observed default rate of 0.2174 for every model. Brier
> scores of 0.149 to 0.152 compare with 0.170 for a model predicting the base
> rate for every loan, an improvement of 10.9% to 12.3%.
>
> Platt scaling produced only marginal changes in Brier score, indicating that
> the models were already well calibrated before adjustment. This is consistent
> with the decision not to resample the training data (Section 3.7): because the
> models were trained on the original class distribution, their predicted
> probabilities already reflect the true base rate, and the cost-derived
> threshold can be applied to them directly. Had SMOTE or class weighting been
> applied, predicted probabilities would have been systematically inflated and
> substantial recalibration would have been required before any cost-based
> threshold could be used.

**Figure 4.2 caption**

> Reliability diagram for the five calibrated models on the test set, using
> eight quantile-based bins. All models track the diagonal closely, confirming
> that predicted probabilities are meaningful in absolute terms. A slight
> positive deviation between 0.15 and 0.35 indicates that risk is marginally
> understated in the region containing the cost-derived threshold p* = 0.1837.

---

## 4.3 Decision thresholds

m (median interest margin) = 0.2250; LGD = 1.0; **p\* = 0.1837**

**Empirically cost-minimising thresholds (validation set)**

| Model | Empirical threshold | Deviation from p* |
|---|---|---|
| Logistic Regression | 0.15 | −0.034 |
| Random Forest | 0.17 | −0.014 |
| XGBoost | 0.19 | +0.006 |
| LightGBM | 0.23 | +0.046 |
| CatBoost | 0.15 | −0.034 |

Mean empirical threshold = 0.178.

**Findings**

- The empirical optima cluster around the theoretical value, which was derived
  from the cost matrix alone with no reference to model output.
- The cost curve is flat between approximately 0.12 and 0.25, so precision in
  setting the threshold matters less than being in the right region.
- Above 0.5 the curve is flat and expensive, because no loan is scored above
  about 0.55.

**Draft text**

> The empirically cost-minimising thresholds on the validation set ranged from
> 0.15 to 0.23, with a mean of 0.178, against a theoretical value of p* = 0.1837
> derived independently from the cost matrix. The correspondence is close, and
> since the theoretical threshold uses no information from the data beyond the
> median interest margin, this provides direct empirical support for Elkan's
> (2001) result on this dataset. The agreement depends on the predicted
> probabilities being calibrated; a model trained on resampled data would not
> exhibit it.

---

## 4.4 Total misclassification cost — primary result

**Table 4.3 — Cost by model and threshold (test set, naira)**

| Model | 0.5 | p* = 0.1837 | Empirical |
|---|---|---|---|
| Logistic Regression | 2,705,750 | 2,062,625 | 2,146,625 |
| Random Forest | 2,546,500 | 2,045,912 | **1,941,450** |
| XGBoost | 2,609,000 | 2,011,162 | 2,035,162 |
| LightGBM | 2,689,500 | 1,986,312 | 2,102,562 |
| CatBoost | 2,544,000 | **1,904,350** | 2,153,450 |

**Benchmarks**

| Policy | Cost | Best model vs it |
|---|---|---|
| Approve every loan | 2,890,000 | −34.1% |
| Reject every loan | 2,331,700 | −18.3% |
| **CatBoost at p\*** | **1,904,350** | — |

Cost per loan at the best result: ₦2,179.

**Normalised cost (1.0 = approve every loan)**

| Model | 0.5 | p* | Empirical |
|---|---|---|---|
| Logistic Regression | 0.9362 | 0.7137 | 0.7428 |
| Random Forest | 0.8811 | 0.7079 | 0.6718 |
| XGBoost | 0.9028 | 0.6959 | 0.7042 |
| LightGBM | 0.9306 | 0.6873 | 0.7275 |
| CatBoost | 0.8803 | **0.6589** | 0.7451 |

**Classification at p\*** — recall 0.642 to 0.711; precision 0.329 to 0.391;
rejections 315 to 410 of 874 (36% to 47%).

**Classification at 0.5** — recall 0.126 to 0.216; rejections 45 to 72.

**Findings**

1. **The theoretical threshold beat the empirically tuned one for four of the
   five models.** Only Random Forest did better with its empirical threshold.
   CatBoost's empirical threshold cost 13% more than its theoretical one on
   test. The empirical threshold is fitted to the validation set and does not
   transfer; p* is derived from the cost matrix and is not subject to that
   overfitting. This is a stronger result than mere agreement between the two.
2. **The conventional 0.5 threshold is close to useless here**, costing 88% to
   94% of the approve-everything policy and catching only 13% to 22% of
   defaults.
3. **Rejecting every loan costs less than approving every loan** under LGD = 1
   (2,331,700 against 2,890,000), which shows how aggressive the total-loss
   assumption is and motivates the sensitivity analysis.
4. **CatBoost wins on precision, not recall.** At p* all five models identify a
   similar number of defaults (122 to 135 of 190), but the number of rejections
   differs sharply:

   | Model | Rejected | TP | Precision |
   |---|---|---|---|
   | CatBoost | 315 | 123 | 0.3905 |
   | XGBoost | 325 | 124 | 0.3815 |
   | Random Forest | 331 | 122 | 0.3686 |
   | Logistic Regression | 379 | 126 | 0.3325 |
   | LightGBM | 410 | 135 | 0.3293 |

   LightGBM catches the most defaults but refuses 410 applicants to do it;
   CatBoost catches 123 while refusing 315 — 95 fewer creditworthy borrowers
   turned away. Since each false positive costs m x L, this efficiency accounts
   for the entire cost advantage. It also makes CatBoost the preferable choice on
   access-to-credit grounds as well as on cost.

   > CatBoost's cost advantage arises from precision rather than recall. At p* it
   > identified 123 of 190 defaults, comparable to the other models, but rejected
   > only 315 applications against 325 to 410 for the others. Since each false
   > positive incurs the forgone interest m x L, refusing fewer creditworthy
   > borrowers for equivalent default detection accounts for the difference in
   > total cost.

5. **Precision at p\* is 0.33 to 0.39.** Of the 315 loans CatBoost rejects,
   roughly 192 would have repaid — about two creditworthy borrowers refused for
   every default prevented. This is what the cost matrix implies when principal
   at risk is 4.4x the interest earned, but it must be addressed in Chapter 5
   and connects to the access-to-finance concern in Section 3.16.

**Draft text**

> The lowest total cost was achieved by CatBoost at the cost-derived threshold:
> ₦1,904,350 on the test set, or ₦2,179 per loan. This is 34.1% below the cost of
> approving every application (₦2,890,000) and 18.3% below the cost of refusing
> every application (₦2,331,700). At the conventional threshold of 0.5 the same
> model cost ₦2,544,000, only 12.0% better than approving every loan, because no
> borrower in the test set received a predicted probability above approximately
> 0.55.
>
> Notably, the theoretically derived threshold outperformed the empirically
> cost-minimising threshold for four of the five models. The empirical threshold
> is fitted to the validation set and does not transfer; p* is derived from the
> cost matrix alone and is therefore not subject to that overfitting.
>
> The improvement in cost is achieved by trading precision for recall. At p* the
> models rejected between 36% and 47% of applications and identified between 64%
> and 71% of defaults, against 5% to 8% of applications and 13% to 22% of
> defaults at the conventional threshold. Precision at p* was between 0.33 and
> 0.39, meaning that roughly two creditworthy borrowers were refused for every
> defaulting borrower correctly identified. This follows directly from the cost
> matrix, in which the principal at risk on an approved bad loan is
> approximately 4.4 times the interest forgone on a rejected good loan, and its
> implications for access to credit are considered in Chapter 5.

---

## 4.5 LGD sensitivity

**Total cost by LGD (test set, naira)**

| Model | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|
| Logistic Regression | 721,250 | 1,267,375 | 1,600,812 | 2,062,625 |
| Random Forest | 690,500 | 1,244,500 | 1,730,312 | 2,045,912 |
| XGBoost | 703,500 | 1,261,500 | 1,741,788 | 2,011,162 |
| LightGBM | 703,500 | 1,288,250 | 1,706,062 | 1,986,312 |
| CatBoost | 669,000 | 1,277,188 | 1,724,312 | 1,904,350 |

**Rejection rate by LGD**

| Model | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|
| Logistic Regression | 0.069 | 0.162 | 0.276 | 0.434 |
| Random Forest | 0.093 | 0.167 | 0.243 | 0.379 |
| XGBoost | 0.080 | 0.153 | 0.229 | 0.372 |
| LightGBM | 0.074 | 0.192 | 0.312 | 0.469 |
| CatBoost | 0.080 | 0.160 | 0.211 | 0.360 |

**Finding.** The aggressive rejection policy is largely an artefact of assuming
total loss of principal. Under partial recovery the model rejects far fewer
applicants — 7% to 9% at LGD = 0.25 — which substantially softens the
access-to-credit concern.

**Draft text**

> Because LGD = 1 assumes total loss of principal, the threshold and resulting
> cost were recalculated across a range of recovery assumptions. The rejection
> rate is highly sensitive to this assumption: at LGD = 1 the models rejected
> between 36% and 47% of applications, but at LGD = 0.25 this fell to between 7%
> and 9%. The stringency of the decision rule is therefore driven substantially
> by the loss assumption rather than by the models themselves. A lender able to
> recover part of the principal on a defaulted loan would operate a markedly less
> restrictive policy under the same framework.

---

## 4.6 Statistical comparison

**DeLong test (test-set ROC-AUC, all pairs)** — no pair significant; smallest
p-value 0.171 (Random Forest vs XGBoost). Largest difference 0.0180
(Logistic Regression vs XGBoost, p = 0.181).

**Friedman test across five cross-validation folds** — χ² = 4.000, p = 0.406.
Not significant, so no post-hoc test performed.

Average ranks (1 = best): CatBoost 2.4, XGBoost 2.6, Random Forest 2.8,
LightGBM 3.0, Logistic Regression 4.2. Largest rank difference 1.800 against a
Nemenyi critical difference of 2.728.

Mean fold AUC varied by approximately 0.059 across folds but only 0.015 across
models.

**Draft text**

> Neither test found significant differences between the models. DeLong's test on
> the test-set ROC-AUCs returned no significant pairwise comparison (all
> p > 0.17). The Friedman test across five cross-validation folds was likewise
> non-significant (χ² = 4.000, p = 0.406), so no post-hoc analysis was performed;
> the largest observed difference in average rank was 1.800, against a Nemenyi
> critical difference of 2.728. Mean fold AUC varied by approximately 0.059
> across folds but by only 0.015 across models, indicating that discrimination
> performance on this dataset depends more on which loans fall into the
> evaluation sample than on the choice of algorithm. Model selection therefore
> rests on total misclassification cost and on interpretability rather than on
> discrimination, since the latter does not distinguish the candidates.

---

## 4.7 Explainability (SHAP)

Explanations are for **CatBoost**, the lowest-cost model, at p* = 0.1837.
TreeExplainer, computed on the 874 test loans, in log-odds units.
Base value E[f(X)] = −1.406, equivalent to a probability of 0.197, close to the
observed base rate of 0.217.

**Table 4.6 — Global feature importance (mean absolute SHAP value)**

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | prev_late_rate | 0.2595 |
| 2 | prev_days_late_mean | 0.1718 |
| 3 | interest_margin | 0.0943 |
| 4 | bank_account_type | 0.0938 |
| 5 | employment_status_clients | 0.0485 |
| 6 | prev_amount_max | 0.0419 |
| 7 | days_since_last_loan | 0.0411 |
| 8 | age_at_approval | 0.0388 |
| 9 | prev_loan_count | 0.0334 |
| 10 | region | 0.0331 |
| 11 | prev_termdays_mean | 0.0319 |
| 12 | prev_amount_mean | 0.0317 |
| 13 | termdays | 0.0270 |
| 14 | bank_name_clients | 0.0217 |
| 15 | level_of_education_clients | 0.0148 |
| 16 | loanamount | 0.0130 |
| 17 | was_referred | 0.0035 |
| 18 | has_demographics | 0.0024 |
| 19 | has_history | 0.0004 |

**Findings**

1. **Repayment history dominates.** `prev_late_rate` alone is 2.75x the next
   feature, and with `prev_days_late_mean` the two exceed the other seventeen
   combined. The beeswarm shows both effects are monotonic: a worse record
   raises predicted risk.
2. **interest_margin ranks third, and encodes the lender's own pricing.** High
   margin maps to higher predicted risk, so the model is partly learning
   SuperLender's prior risk assessment rather than forming an independent one.
   This needs explicit handling because m also appears in the cost matrix: m is
   known at origination, so using it as a predictor is legitimate, and its role
   in the cost matrix is as a valuation of forgone revenue rather than as a risk
   signal. State this rather than leaving the question open.
3. **The protected attributes sit mid-table** — age_at_approval 8th (0.0388),
   region 10th (0.0331). Non-trivial but well below the repayment features. This
   is the payoff from including them as model inputs (Section 3.13.2): their
   influence is visible and measurable. Modest global influence does not rule out
   disparate outcomes, which Section 4.8 tests.
4. **The three indicator flags are essentially inert** — has_history 0.0004,
   has_demographics 0.0024, was_referred 0.0035. Missingness itself carries
   almost no signal. Section 3.5.1 justified has_demographics on the grounds that
   it might; the honest finding is that it did not.

**Local explanations — four test cases**

| Case | Row | p (calibrated) | f(x) log-odds | Actual |
|---|---|---|---|---|
| Correctly identified bad loan | 698 | 0.8000 | +1.048 | 1 |
| Correctly approved good loan | 585 | 0.0994 | −2.626 | 0 |
| Missed bad loan (false negative) | 438 | 0.1077 | −2.387 | 1 |
| Borderline case near p* | 386 | 0.1838 | −1.380 | 1 |

- **Correctly identified bad loan.** `prev_days_late_mean` +0.8 and
  `prev_late_rate` +0.64 account for almost the entire departure from the base
  value. A visibly poor repayment record, flagged for the right reason.
- **Correctly approved good loan.** Every contribution negative, led by
  `bank_account_type` −0.23 and the two repayment features.
- **Missed bad loan — the most informative case.** Every contribution is
  negative. The borrower looked good on every dimension the model can observe:
  clean repayment history, favourable account type, no warning signs. The loan
  still defaulted. This illustrates the ceiling on history-based prediction
  rather than a defect in the model, and it is the substantive answer to why
  recall at p* is 0.65 rather than higher.
- **Borderline case.** `prev_late_rate` −0.20 pushed towards approval while
  `interest_margin` +0.12 pushed back, leaving the prediction almost exactly at
  the base value and 0.0001 above the threshold. The loan defaulted. It shows
  conflicting signals resolving at the decision boundary, and `interest_margin`
  carrying information the repayment history missed.

**Draft text**

> Feature attributions were computed for CatBoost, the model achieving the lowest
> total cost, using TreeExplainer on the 874 test loans. Repayment history
> dominated the global ranking: the share of prior loans repaid late
> (mean |SHAP| = 0.2595) and the mean days between first due date and first
> repayment (0.1718) together contributed more than the remaining seventeen
> features combined. The direction of both effects was monotonic, with a worse
> prior record raising predicted risk.
>
> The interest margin ranked third (0.0943), with higher-priced loans receiving
> higher predicted risk. This reflects the lender's own risk-based pricing: the
> model partly recovers SuperLender's prior assessment of each borrower rather
> than forming an entirely independent one. The margin is known at origination
> and is therefore a legitimate predictor; its separate role in the cost matrix
> is as a valuation of forgone revenue on a rejected good loan, not as a risk
> signal, so no circularity arises.
>
> The two protected attributes ranked eighth (age at approval, 0.0388) and tenth
> (region, 0.0331). Their influence is modest relative to repayment history but
> not negligible, which is the outcome the decision to include them as model
> inputs (Section 3.13.2) was intended to make visible. Whether this influence
> translates into materially different treatment across groups is examined in
> Section 4.8.
>
> The three missingness indicators contributed almost nothing (has_history
> 0.0004, has_demographics 0.0024, was_referred 0.0035). Section 3.5.1 retained
> has_demographics on the grounds that the absence of a demographic record might
> itself be informative; the evidence here is that it is not.
>
> Local explanations were produced for four contrasting test cases. [Describe the
> four waterfalls, particularly the false negative.] The false negative is
> instructive: every feature contribution for that borrower was negative, meaning
> the loan appeared low-risk on every dimension the model can observe, yet it
> defaulted. This illustrates the limit of prediction from historical repayment
> behaviour — a borrower deteriorating for the first time leaves no trace in the
> features available — rather than a deficiency in the model itself.

**Figure captions**

> **Figure 4.5:** Global feature importance for CatBoost, measured as the mean
> absolute SHAP value across the 874 test loans.

> **Figure 4.6:** Distribution of SHAP values by feature. Colour indicates the
> feature value, from low (blue) to high (red). Categorical features are shown in
> grey as they have no natural ordering.

> **Figure 4.7:** Local explanations for four test cases. Values are in log-odds
> units of the uncalibrated model; the calibrated probability is given in each
> title. E[f(X)] = −1.406 is the model's average output, equivalent to a
> probability of 0.197.

---

## 4.8 Fairness audit

CatBoost at p* = 0.1837, test set. All groups exceeded the minimum size of 30,
so none were excluded.

**Table 4.7 — Fairness metrics by geographic region**

| Group | n | Defaults | Rejection rate | TPR | FPR |
|---|---|---|---|---|---|
| South West | 434 | 91 | 0.3664 | 0.6154 | 0.3003 |
| Unknown | 233 | 54 | 0.3519 | 0.6667 | 0.2570 |
| North Central | 92 | 15 | 0.3370 | 0.7333 | 0.2597 |
| South South | 77 | 17 | 0.3377 | 0.7059 | 0.2333 |
| Other regions | 38 | 13 | 0.4474 | 0.6154 | 0.3600 |

**Table 4.8 — Fairness metrics by age band**

| Group | n | Defaults | Rejection rate | TPR | FPR |
|---|---|---|---|---|---|
| 25-34 | 387 | 81 | 0.3463 | 0.6420 | 0.2680 |
| Unknown | 228 | 53 | 0.3553 | 0.6604 | 0.2629 |
| 35+ | 218 | 45 | 0.3394 | 0.6444 | 0.2601 |
| **18-24** | **41** | **11** | **0.6341** | 0.6364 | **0.6333** |

**Summary differences**

| Metric | region_fair | age_band |
|---|---|---|
| Demographic parity difference | 0.1104 | 0.2947 |
| Disparate impact ratio | **0.8335** | **0.5539** |
| Equal opportunity difference | 0.1179 | 0.0240 |
| Equalised odds difference | 0.1267 | 0.3732 |

Four-fifths rule: a disparate impact ratio below 0.80 indicates potential
adverse impact (Feldman et al., 2015).

### Findings

**1. Region passes; age fails.** The disparate impact ratio for region is 0.8335,
above the four-fifths threshold. For age it is 0.5539, well below.

**2. The age disparity is entirely in false positives, not true positives.**
TPR is near-identical across all four age bands (0.636 to 0.660), giving an equal
opportunity difference of just 0.0240. FPR for 18-24 is 0.6333, against
approximately 0.26 for every other band — a factor of 2.4. Nineteen of the
thirty creditworthy 18-24 applicants were refused.

This is the empirical justification for auditing on equalised odds rather than
equal opportunity (Section 3.13.5). An audit using equal opportunity alone would
have found this model fair on age.

**3. The mechanism is a threshold effect, not a scoring bias.**

| Band | Median p | 25th pct | 75th pct | Share above p* | Actual default rate |
|---|---|---|---|---|---|
| 18-24 | **0.196** | 0.158 | 0.238 | 0.634 | 0.268 |
| 25-34 | **0.158** | 0.136 | 0.217 | 0.346 | 0.209 |
| 35+ | **0.159** | 0.143 | 0.203 | 0.339 | 0.206 |

p* = 0.1837 falls **between** the medians. The median 18-24 borrower is rejected;
the median older borrower is approved. The interquartile ranges overlap heavily,
so these are near-identical distributions shifted by approximately 0.037, cut at
the point where that shift has maximum effect.

**4. The risk difference is real but the rejection difference is
disproportionate.** 18-24 borrowers do default more often (26.8% against 20.6%
for 35+, a ratio of 1.30) and have a worse prior late rate (0.222 against 0.173).
But the rejection ratio is 1.87 — roughly 44% greater than their relative risk
warrants.

**5. The model slightly under-predicts risk for young borrowers.** Mean predicted
probability for 18-24 is 0.235 against an observed default rate of 0.268; for
35+ it is 0.207 against 0.206. So young borrowers are not being over-scored —
they are marginally under-scored and still rejected at twice the rate.

**6. Caveat on group size.** The 18-24 band has 41 test loans, 11 of them
defaults. Above the stated minimum of 30, but the FPR estimate rests on 30
non-defaulting borrowers. The effect is large enough to be unlikely to be noise,
but the precision of the estimate should be stated.

**Draft text**

> The fairness audit found no material disparity across geographic region: the
> disparate impact ratio was 0.8335, above the four-fifths threshold, with
> rejection rates ranging from 0.337 to 0.447 across the five reporting groups.
>
> Age presented a different picture. The disparate impact ratio was 0.5539, well
> below the four-fifths threshold, driven by the 18-24 band, which was rejected
> at 0.634 against approximately 0.34 for every other band. Critically, this
> disparity was confined to false positives. True positive rates were
> near-identical across all four age bands (0.636 to 0.660), giving an equal
> opportunity difference of 0.0240, while the false positive rate for 18-24
> borrowers was 0.6333 against approximately 0.26 elsewhere. Nineteen of the
> thirty creditworthy applicants aged 18 to 24 were refused credit. An audit
> conducted on equal opportunity alone would have found this model fair; the
> decision to measure equalised odds (Section 3.13.5) was necessary to detect the
> disparity.
>
> Examination of the predicted probability distributions shows that this arises
> as a threshold effect rather than a scoring bias. The median predicted
> probability was 0.196 for the 18-24 band and 0.158 for the 25-34 band, so the
> cost-derived threshold of 0.1837 falls between them: the median young borrower
> is rejected and the median older borrower approved. The interquartile ranges
> overlap substantially, indicating two near-identical distributions separated by
> approximately 0.037 and divided at the point where that separation has greatest
> effect.
>
> The underlying difference in risk is real but smaller than the difference in
> treatment. Borrowers aged 18 to 24 defaulted at 26.8% against 20.6% for those
> aged 35 and over, a ratio of 1.30, while their rejection rate was 1.87 times
> higher. Moreover, the model marginally under-predicted risk for this group
> (mean predicted probability 0.235 against an observed rate of 0.268), so the
> disparity cannot be attributed to inflated risk scores.
>
> This finding has implications beyond the present model. Applying a single
> global threshold to groups whose risk distributions differ slightly will
> produce disproportionate rejection wherever the threshold falls within the dense
> region of those distributions, even when the underlying probabilities are well
> calibrated for every group. The effect is a property of cost-sensitive decision
> rules in general rather than of this classifier.

---

## 4.9 Mitigation and the fairness-cost frontier

Baseline: CatBoost at p* = 0.1837, cost ₦1,904,350.
ThresholdOptimizer fitted on the validation set, applied unchanged to the test
set.

**Table 4.9 — Post-processing mitigation**

| Attribute | Constraint | Rejected | Cost after | Increase | Increase % | DP diff after | EO diff after |
|---|---|---|---|---|---|---|---|
| region_fair | equalised odds | 12 | 2,858,000 | 953,650 | +50.1% | 0.0263 | 0.0769 |
| region_fair | TPR parity | 48 | 2,646,500 | 742,150 | +39.0% | 0.0737 | 0.2088 |
| age_band | equalised odds | 71 | 2,489,000 | 584,650 | +30.7% | 0.0686 | 0.2716 |
| age_band | TPR parity | 70 | 2,517,500 | 613,150 | +32.2% | 0.0468 | 0.1930 |

### Findings

**1. Mitigation is expensive.** Every configuration increased total cost by 31%
to 50%.

**2. The region/equalised-odds result is degenerate.** It rejects 12 loans out of
874 and costs ₦2,858,000, within 1% of the ₦2,890,000 cost of approving every
application. It achieved fairness by abandoning discrimination altogether. This
should be reported as such rather than presented as a viable operating point.

**3. Important limitation.** ThresholdOptimizer was configured with
`objective='accuracy_score'`, so it selected group thresholds maximising accuracy
rather than minimising cost. These figures therefore overstate the true price of
fairness; a cost-aware post-processing method would lie lower on the frontier.
This is a genuine limitation and a natural direction for future work.

**Draft text**

> Post-processing mitigation was applied using Fairlearn's ThresholdOptimizer,
> fitted on the validation set and applied unchanged to the test set. Four
> configurations were evaluated, combining each protected attribute with each of
> two constraints.
>
> All four substantially reduced the measured disparity but at considerable cost.
> The increase in total misclassification cost ranged from 30.7% to 50.1%. The
> configuration applying equalised odds across geographic regions achieved the
> lowest demographic parity difference (0.0263) but did so by rejecting only 12
> of 874 applications, at a cost of ₦2,858,000 — within 1% of the cost of
> approving every application. This configuration satisfies the fairness
> constraint by abandoning discrimination entirely and is not a viable operating
> point.
>
> Two qualifications apply. First, ThresholdOptimizer was configured to maximise
> accuracy rather than to minimise cost, so these figures overstate the price of
> fairness; a cost-aware post-processing method would achieve better positions on
> the frontier, and this is left to future work. Second, as argued in Section
> 3.13.5, group-specific thresholds constitute explicit differential treatment on
> the basis of age or region, which may itself be impermissible under fair-lending
> frameworks. These results are therefore presented as a measurement of the
> fairness-cost trade-off available to a lender, not as a deployable
> recommendation.

**Figure caption**

> **Figure 4.8:** Rejection rate and false positive rate by age band at the
> cost-derived threshold. True positive rates are comparable across bands while
> the false positive rate for borrowers aged 18 to 24 is approximately 2.4 times
> that of other groups.

---

## Corrections made during analysis — worth noting in the chapter or appendix

1. **referredby** was initially one-hot encoded, producing 395 encoded features
   from 21 raw ones, because it holds a referrer identifier rather than a
   category. Reduced to a binary indicator `was_referred`; encoded feature count
   fell to 55. CatBoost's validation log loss improved from 0.45451 to 0.44967.
2. **bank_branch_clients** (99.24% missing, 31 distinct values where present)
   and **totaldue** (redundant given loanamount and interest_margin) were
   dropped.
3. **Isotonic calibration was replaced by Platt scaling.** Isotonic appeared
   superior on validation but was fitting the same data it was measured on; it
   raised validation AUC, which a monotonic transform cannot legitimately do.
4. **An early Friedman test gave inflated fold AUCs** (Random Forest 0.8607)
   because scikit-learn's FrozenEstimator does not refit when cloned, so models
   fitted on the full training set were scored on their own training data. Fixed
   by rebuilding fresh estimators from the saved hyperparameters; fold AUCs fell
   to 0.690–0.705, consistent with the test set.

---

## Chapter 3 edits still outstanding

- 3.5.3: GPS points outside Nigeria — change 31 to **36** (the bounding-box check
  gives 31; the GADM point-in-polygon join, which actually assigned the Unknown
  region, gives 36).
- 3.5.1: "1,099(25%)" → 25.16%.
- 3.5.2: add a sentence on `referredby` being reduced to a binary indicator.
- 3.6.2: replace the `[name source]` placeholder with the GADM citation; add the
  collapsed-region justification.
- Table 3.3: add `was_referred`, `has_history`, `region_fair`, `age_band`; remove
  `bank_branch_clients` and `totaldue` if listed; amend the `region` row to "one
  of six geopolitical zones".
- New section for the train/validation/test split, between 3.5.3 and encoding;
  renumber encoding to 3.5.5.
- **Table 3.5 is used twice** — LGD sensitivity (3.8.4) and hyperparameter search
  spaces (3.9.3). Renumber one, and everything after it.
- 3.7: remove the brackets around 3.59.
- 3.8.1: the cost matrix has lost its multiplication signs and the bottom-right
  cell reads "01" instead of 0; the paragraph below reads "A false negative ( )".
- 3.9.3: state Optuna, 50 trials per model, and log loss as the objective.
- 3.9.4: state Platt scaling and why isotonic was rejected.
- 3.13.1: age bands and region groups contradict 3.6.2 — must match
  18-24 / 25-34 / 35+ / Unknown and the five region groups.
- 3.13.5: the ThresholdOptimizer mechanism sentence was lost; restore it.
- 3.15: Python 3.13.7, VS Code with the Jupyter extension, the filled library
  table, i7-10750H / 32 GB, repository link.
- DeLong citation: `(DeLong, DeLong & Clarke-Pearson, 1988)` → `(DeLong et al.,
  1988)` for APA 7th.
- CatBoost tuning took 8,199 seconds single-threaded, 5,326 seconds with
  `thread_count=-1`. Worth recording in 3.15.

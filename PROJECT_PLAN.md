# Loan Approval Machine-Learning Project Plan

## Purpose

Develop a responsible, explainable, and maintainable loan approval risk model. The model should support lending decisions rather than operate without appropriate policy, compliance, and human oversight.

## Status convention

- `[TODO]`: Work has not started or is still in progress.
- `[DONE]`: Work is complete and has been reviewed.
- `[BLOCKED]`: Work cannot continue until a documented dependency is resolved.

Update the status and notes in this document as each step progresses.

## 1. Build baseline models

- [DONE] Define a simple rule-based or scorecard benchmark.
- [DONE] Build an interpretable logistic-regression baseline.
- [DONE] Consider a decision-tree baseline for comparison.
- [DONE] Consider a gradient-boosted model only if it provides a justified improvement.
- [DONE] Record model assumptions, features, parameters, and reproducibility details.
- [DONE] Prefer the simplest model that meets performance, fairness, stability, and explainability requirements.

## 2. Prepare the data

- [DONE] Validate required fields, formats, ranges, and allowed values.
- [DONE] Identify duplicates and inconsistent records.
- [DONE] Define an explicit approach for missing values.
- [DONE] Detect target leakage and proxy variables.
- [DONE] Examine class imbalance and document any treatment applied.
- [DONE] Create chronological training, validation, and test sets where possible.
- [DONE] Define a reproducible preprocessing workflow.

## 3. Define the decision

- [DONE] Define the prediction target, such as default within 12 months.
- [DONE] Decide whether the output is a probability of default, risk score, or decision-support recommendation.
- [DONE] Define the model's role in approval, decline, and manual-review decisions.
- [DONE] Establish measurable success criteria for risk, approval rate, profitability, fairness, and decision speed.
- [DONE] Document what is explicitly outside the model's scope.

## 4. Establish compliance and governance

- [DONE] Identify applicable lending, privacy, consumer-protection, and fair-lending requirements.
- [DONE] Define prohibited decision attributes and restricted data uses.
- [DONE] Determine whether protected attributes may be retained separately for fairness testing.
- [DONE] Define explainability and adverse-action notice requirements.
- [DONE] Establish human-review, override, and applicant-appeal processes.
- [DONE] Assign owners for model development, validation, compliance approval, and production operation.
- [DONE] Create a model risk and limitation register.

## 5. Collect and understand data

- [DONE] Inventory all data sources and confirm permission to use them.
- [DONE] Create a data dictionary with definitions, types, units, and ownership.
- [DONE] Confirm that every predictor was available at application time.
- [DONE] Define the outcome label and observation window precisely.
- [DONE] Assess data coverage, quality, freshness, and representativeness.
- [DONE] Investigate selection bias when outcomes exist only for previously approved applicants.
- [DONE] Establish data lineage and versioning expectations.

## 6. Perform exploratory analysis

- [DONE] Analyze feature and target distributions.
- [DONE] Review default rates and historical approval patterns.
- [DONE] Examine correlations, missingness, outliers, and unusual values.
- [DONE] Analyze changes in applicant populations and outcomes over time.
- [DONE] Compare relevant measures across demographic groups where legally permitted.
- [DONE] Identify signs of historical bias or unstable relationships.
- [DONE] Document findings that affect modeling or policy decisions.

## 7. Evaluate the model

- [DONE] Define evaluation metrics before final model selection.
- [DONE] Measure ROC-AUC and precision-recall AUC.
- [DONE] Review precision, recall, and confusion matrices at relevant thresholds.
- [DONE] Measure calibration and Brier score.
- [DONE] Compare default and approval rates across risk-score bands.
- [DONE] Estimate expected financial loss and other business outcomes.
- [DONE] Compare candidate models with the existing lending process and baseline models.
- [DONE] Evaluate performance only on held-out data not used during training or tuning.

## 8. Test fairness and robustness

- [DONE] Define fairness measures with compliance and policy stakeholders.
- [DONE] Compare approval rates, errors, calibration, and outcomes across relevant groups.
- [DONE] Investigate features that may act as proxies for protected attributes.
- [DONE] Test applicants near proposed decision thresholds.
- [DONE] Test missing, invalid, unusual, and extreme input values.
- [DONE] Run economic stress and population-shift scenarios.
- [DONE] Define acceptable performance and fairness tolerances.
- [DONE] Document mitigations, residual risks, and approval decisions.

## 9. Select decision thresholds

- [DONE] Define low-, medium-, and high-risk decision bands.
- [DONE] Specify when an application requires manual review or more information.
- [DONE] Evaluate thresholds against risk tolerance, expected loss, operational capacity, and fairness results.
- [DONE] Separate policy rules from model predictions where appropriate.
- [DONE] Document threshold ownership and the approval process for future changes.

## 10. Add explanations and safeguards

- [DONE] Define valid reason codes tied directly to model inputs and decision logic.
- [DONE] Validate that explanations are accurate, consistent, and understandable.
- [DONE] Record the model version, data version, score, threshold, and decision for each application.
- [DONE] Log manual reviews and overrides with reasons.
- [DONE] Define safe behavior for missing, invalid, or unavailable inputs.
- [DONE] Establish audit, access-control, retention, and privacy requirements.

## 11. Deploy safely

- [DONE] Package preprocessing and prediction as one versioned workflow.
- [DONE] Validate production inputs against the approved data contract.
- [DONE] Run the model in shadow mode without affecting decisions.
- [DONE] Conduct a limited pilot with human review.
- [DONE] Define acceptance criteria for controlled production rollout.
- [DONE] Create rollback and incident-response procedures.
- [DONE] Obtain model validation, compliance, business, and operational approvals.

## 12. Monitor and maintain

- [DONE] Monitor approval rates, default rates, calibration, and prediction quality.
- [DONE] Monitor data quality, feature drift, and concept drift.
- [DONE] Monitor fairness measures and group-level outcomes.
- [DONE] Review manual overrides, applicant appeals, and operational failures.
- [DONE] Define alert thresholds and responsible owners.
- [DONE] Define conditions for rollback, recalibration, retraining, and retirement.
- [DONE] Schedule periodic independent validation and governance review.
- [DONE] Maintain a change log for data, models, thresholds, and policy updates.

## Current milestone

- [DONE] Complete Step 1 and establish the baseline results that later models and process changes will be measured against.

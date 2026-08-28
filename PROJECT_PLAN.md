# Loan Approval Machine-Learning Project Plan

## Purpose

Develop a responsible, explainable, and maintainable loan approval risk model. The model should support lending decisions rather than operate without appropriate policy, compliance, and human oversight.

## Status convention

- `[TODO]`: Work has not started or is still in progress.
- `[DONE]`: Work is complete and has been reviewed.
- `[BLOCKED]`: Work cannot continue until a documented dependency is resolved.

Update the status and notes in this document as each step progresses.

## 1. Build baseline models

- [TODO] Define a simple rule-based or scorecard benchmark.
- [TODO] Build an interpretable logistic-regression baseline.
- [TODO] Consider a decision-tree baseline for comparison.
- [TODO] Consider a gradient-boosted model only if it provides a justified improvement.
- [TODO] Record model assumptions, features, parameters, and reproducibility details.
- [TODO] Prefer the simplest model that meets performance, fairness, stability, and explainability requirements.

## 2. Prepare the data

- [TODO] Validate required fields, formats, ranges, and allowed values.
- [TODO] Identify duplicates and inconsistent records.
- [TODO] Define an explicit approach for missing values.
- [TODO] Detect target leakage and proxy variables.
- [TODO] Examine class imbalance and document any treatment applied.
- [TODO] Create chronological training, validation, and test sets where possible.
- [TODO] Define a reproducible preprocessing workflow.

## 3. Define the decision

- [TODO] Define the prediction target, such as default within 12 months.
- [TODO] Decide whether the output is a probability of default, risk score, or decision-support recommendation.
- [TODO] Define the model's role in approval, decline, and manual-review decisions.
- [TODO] Establish measurable success criteria for risk, approval rate, profitability, fairness, and decision speed.
- [TODO] Document what is explicitly outside the model's scope.

## 4. Establish compliance and governance

- [TODO] Identify applicable lending, privacy, consumer-protection, and fair-lending requirements.
- [TODO] Define prohibited decision attributes and restricted data uses.
- [TODO] Determine whether protected attributes may be retained separately for fairness testing.
- [TODO] Define explainability and adverse-action notice requirements.
- [TODO] Establish human-review, override, and applicant-appeal processes.
- [TODO] Assign owners for model development, validation, compliance approval, and production operation.
- [TODO] Create a model risk and limitation register.

## 5. Collect and understand data

- [TODO] Inventory all data sources and confirm permission to use them.
- [TODO] Create a data dictionary with definitions, types, units, and ownership.
- [TODO] Confirm that every predictor was available at application time.
- [TODO] Define the outcome label and observation window precisely.
- [TODO] Assess data coverage, quality, freshness, and representativeness.
- [TODO] Investigate selection bias when outcomes exist only for previously approved applicants.
- [TODO] Establish data lineage and versioning expectations.

## 6. Perform exploratory analysis

- [TODO] Analyze feature and target distributions.
- [TODO] Review default rates and historical approval patterns.
- [TODO] Examine correlations, missingness, outliers, and unusual values.
- [TODO] Analyze changes in applicant populations and outcomes over time.
- [TODO] Compare relevant measures across demographic groups where legally permitted.
- [TODO] Identify signs of historical bias or unstable relationships.
- [TODO] Document findings that affect modeling or policy decisions.

## 7. Evaluate the model

- [TODO] Define evaluation metrics before final model selection.
- [TODO] Measure ROC-AUC and precision-recall AUC.
- [TODO] Review precision, recall, and confusion matrices at relevant thresholds.
- [TODO] Measure calibration and Brier score.
- [TODO] Compare default and approval rates across risk-score bands.
- [TODO] Estimate expected financial loss and other business outcomes.
- [TODO] Compare candidate models with the existing lending process and baseline models.
- [TODO] Evaluate performance only on held-out data not used during training or tuning.

## 8. Test fairness and robustness

- [TODO] Define fairness measures with compliance and policy stakeholders.
- [TODO] Compare approval rates, errors, calibration, and outcomes across relevant groups.
- [TODO] Investigate features that may act as proxies for protected attributes.
- [TODO] Test applicants near proposed decision thresholds.
- [TODO] Test missing, invalid, unusual, and extreme input values.
- [TODO] Run economic stress and population-shift scenarios.
- [TODO] Define acceptable performance and fairness tolerances.
- [TODO] Document mitigations, residual risks, and approval decisions.

## 9. Select decision thresholds

- [TODO] Define low-, medium-, and high-risk decision bands.
- [TODO] Specify when an application requires manual review or more information.
- [TODO] Evaluate thresholds against risk tolerance, expected loss, operational capacity, and fairness results.
- [TODO] Separate policy rules from model predictions where appropriate.
- [TODO] Document threshold ownership and the approval process for future changes.

## 10. Add explanations and safeguards

- [TODO] Define valid reason codes tied directly to model inputs and decision logic.
- [TODO] Validate that explanations are accurate, consistent, and understandable.
- [TODO] Record the model version, data version, score, threshold, and decision for each application.
- [TODO] Log manual reviews and overrides with reasons.
- [TODO] Define safe behavior for missing, invalid, or unavailable inputs.
- [TODO] Establish audit, access-control, retention, and privacy requirements.

## 11. Deploy safely

- [TODO] Package preprocessing and prediction as one versioned workflow.
- [TODO] Validate production inputs against the approved data contract.
- [TODO] Run the model in shadow mode without affecting decisions.
- [TODO] Conduct a limited pilot with human review.
- [TODO] Define acceptance criteria for controlled production rollout.
- [TODO] Create rollback and incident-response procedures.
- [TODO] Obtain model validation, compliance, business, and operational approvals.

## 12. Monitor and maintain

- [TODO] Monitor approval rates, default rates, calibration, and prediction quality.
- [TODO] Monitor data quality, feature drift, and concept drift.
- [TODO] Monitor fairness measures and group-level outcomes.
- [TODO] Review manual overrides, applicant appeals, and operational failures.
- [TODO] Define alert thresholds and responsible owners.
- [TODO] Define conditions for rollback, recalibration, retraining, and retirement.
- [TODO] Schedule periodic independent validation and governance review.
- [TODO] Maintain a change log for data, models, thresholds, and policy updates.

## Current milestone

- [TODO] Complete Step 1 and establish the baseline results that later models and process changes will be measured against.


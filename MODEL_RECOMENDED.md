# Recommended Classification Models

## Purpose

This document identifies machine-learning classifiers that can produce a score or probability for a loan approval decision-support system. The preferred prediction is a probability of default within a clearly defined period. Lending policy should convert that risk estimate into approval, manual-review, or decline decisions.

## Recommended models

| Model | Output | Explainability | Recommended use |
| --- | --- | --- | --- |
| Logistic regression | Probability | High | Primary baseline |
| Credit scorecard | Points and probability | Very high | Transparent lending model |
| Decision tree | Probability | High | Interpretable nonlinear baseline |
| Random forest | Probability | Medium | Performance comparison |
| Gradient-boosted trees | Probability or score | Medium | Advanced performance candidate |
| Neural network | Probability | Low | Consider only with sufficient justification and data |
| Support vector machine | Decision score; calibrated probability | Low to medium | Lower-priority comparison |
| Naive Bayes | Probability | Medium | Simple benchmark only |

## 1. Logistic regression

Logistic regression is the recommended starting model.

- Produces a probability between 0 and 1.
- Offers coefficients that help explain how features influence estimated risk.
- Is relatively easy to validate, monitor, and govern.
- Can support standardized decision-reason generation.
- May require feature transformations and probability calibration.

Example model output:

- Probability of default: `0.12`
- Probability of repayment: `0.88`
- Risk category: low

## 2. Credit scorecard

A credit scorecard commonly converts a logistic-regression result into an applicant risk score.

- Produces an understandable points-based score.
- Can map each score to an estimated probability of default.
- Supports transparent lending rules and reason codes.
- Requires carefully designed feature binning and score construction.

## 3. Decision tree

- Produces probabilities based on the training observations in each terminal leaf.
- Captures nonlinear relationships and feature interactions.
- Is relatively easy to visualize and explain.
- Can overfit or become unstable without appropriate constraints.
- May require probability calibration.

## 4. Random forest

- Combines predictions from multiple decision trees.
- Produces class probabilities.
- Handles nonlinear relationships and feature interactions.
- Is generally more stable than a single decision tree.
- Is less transparent than logistic regression or a scorecard.
- Requires evaluation of probability calibration.

## 5. Gradient-boosted trees

Gradient-boosted trees are the recommended advanced model candidate after the baseline has been established.

- Often perform well with structured financial data.
- Produce probabilities or decision scores.
- Capture nonlinear relationships and complex interactions.
- Require careful tuning, validation, and explanation.
- Frequently benefit from probability calibration.

Possible implementations include histogram-based gradient boosting, XGBoost, and LightGBM. A new dependency should be introduced only if the existing project environment cannot meet the requirement.

## 6. Neural network

- Can produce a probability through an appropriate output layer.
- Can represent complex relationships.
- Usually requires more data and validation effort.
- Is harder to explain, govern, and monitor.
- Should be considered only when it provides a material, validated benefit over simpler models.

## 7. Support vector machine

- Produces a decision score by default.
- Can produce probabilities when a calibration method is applied.
- May work well on some datasets but can be difficult to explain and scale.
- Is a lower-priority candidate for the initial loan model.

## 8. Naive Bayes

- Produces class probabilities.
- Is simple and computationally efficient.
- Relies on assumptions that may not fit correlated financial features.
- Is suitable as a benchmark rather than the expected production model.

## Probability calibration

A classification score is not automatically a reliable probability. Calibration should be evaluated on held-out data. If required, candidate methods include:

- Sigmoid or Platt calibration
- Isotonic regression

A well-calibrated model that assigns approximately 20% default risk should show an observed default rate near 20% among comparable applications over the defined outcome period.

## Recommended evaluation set

The initial project should compare:

1. Logistic regression as the primary baseline
2. Credit scorecard as the transparent lending model
3. Decision tree as the interpretable nonlinear baseline
4. Gradient-boosted trees as the advanced performance candidate

Model selection should consider:

- Discrimination, including ROC-AUC and precision-recall AUC
- Probability calibration and Brier score
- Expected financial loss and operational impact
- Fairness across relevant applicant groups
- Stability over time and under economic stress
- Explainability and adverse-action reason requirements
- Implementation, governance, and monitoring complexity

The selected model should provide a risk estimate. Approval thresholds and policy rules should be defined, reviewed, and governed separately from the model prediction.


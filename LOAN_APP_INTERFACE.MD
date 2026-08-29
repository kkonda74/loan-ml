# Loan Application Interface

## Purpose

`loan_app.py` is a command-line demonstration that trains a loan-approval classifier from historical application data and then scores new applicants entered interactively.

The model target is the historical `approved` column. Its output represents patterns in previous approval decisions—not a validated probability of repayment or default. Real lending use requires an outcome-based target, independent validation, fair-lending and legal review, governed thresholds, monitoring, and human oversight.

## Run the application

Use Python 3.11.9 or newer. From the project directory containing `loan_app.py` and `data`, run:

```powershell
python loan_app.py
```

The working directory matters because the data path is relative:

```python
DATA_PATH = Path("data") / "loan_applications.csv"
```

Running from another directory may cause `FileNotFoundError`.

## Application workflow

When executed directly, `loan_app.py` performs five stages:

1. Load and clean historical application data.
2. Validate schema and data quality.
3. Create stratified training, validation, and test sets.
4. Train and evaluate a logistic-regression credit scorecard.
5. Prompt for new applicants and display predictions.

Importing `loan_app` does not run these stages because executable behavior is protected by:

```python
if __name__ == "__main__":
```

## Dataset contract

The source is `data/loan_applications.csv`.

| Column | Role | Expected content |
| --- | --- | --- |
| `applicant_id` | Identifier | Unique application identifier; excluded from modeling |
| `fico_score` | Numeric feature | FICO score; missing values currently become `0` |
| `annual_income` | Numeric feature | Non-negative annual income |
| `loan_amount` | Numeric feature | Positive requested amount |
| `loan_term_months` | Numeric feature | Positive term in months |
| `employment_status` | Categorical feature | `Salaried` or `Self-employed` |
| `years_employed` | Numeric feature | Non-negative employment duration |
| `savings_balance` | Numeric feature | Non-negative savings balance |
| `approved` | Target | `Approved` becomes `1`; other values become `0` |

The ordered model schema is defined by `INPUT_COLUMNS`. New prediction DataFrames must contain every field in that list.

## Stage 1: Data loading and cleaning

The application loads the CSV with pandas and prints column types before and after cleaning. It then:

- Removes `$`, commas, and apostrophes from `annual_income`, `loan_amount`, and `savings_balance`.
- Converts those currency-like columns to numeric values.
- Converts unparseable currency values to missing values.
- Removes records with negative `annual_income`.
- Converts `fico_score` to numeric.
- Replaces missing FICO values with `0`.
- Converts FICO values to `int64`.
- Removes duplicate `applicant_id` records, keeping the first occurrence.

## Stage 2: Data-quality validation

Before training, the script verifies:

- All required identifier, target, and input columns exist.
- At least `MINIMUM_TRAINING_ROWS` rows remain.
- No duplicate applicant IDs remain.
- Feature missingness does not exceed `MAXIMUM_MISSING_RATE`.
- The target has no missing values.
- Each target class has at least `MINIMUM_CLASS_COUNT` records.
- Income, employment duration, and savings are non-negative.
- Loan amount and loan term are greater than zero.

The results are stored in `data_quality_report`. If checks fail, one `ValueError` lists the detected violations and training stops.

## Stage 3: Dataset splitting

The target is encoded as follows:

```python
target = loan_applications[TARGET_COLUMN].eq(POSITIVE_CLASS).astype("int8")
```

Exactly `Approved` becomes `1`; other non-missing labels become `0`.

Two stratified splits produce:

- 60% training data
- 20% validation data
- 20% test data

`RANDOM_STATE` makes the split reproducible. The validation set is summarized but is not currently used for tuning or model selection.

## Stage 4: Model training

The command-line application trains `credit_scorecard`, which combines preprocessing and `LogisticRegression` in one scikit-learn `Pipeline`.

Processing flow:

```text
Raw application fields
    -> numeric and categorical preprocessing
    -> logistic-regression classifier
    -> approval class and probability
```

Numeric features receive median imputation and standardization. Employment status receives most-frequent imputation and one-hot encoding. Packaging these transformations with the classifier ensures new applicants receive the transformations fitted only on training data.

Training prints:

- Accuracy
- ROC-AUC
- Confusion matrix
- Classification report

The file also defines an optional `random_forest` interface, but the current main workflow does not instantiate it.

## Stage 5: Interactive scoring

The application requests these values in order:

1. FICO score
2. Annual income
3. Loan amount
4. Loan term in months
5. Employment selection
6. Years employed
7. Savings balance

For employment, enter `1` for `Salaried`; every other numeric value maps to `Self-employed`.

`read_float` handles each prompt. If conversion fails, it prints an error, stops the current application entry, and proceeds to the continue/quit question.

The application converts the responses into a one-row DataFrame and sends the original fields to the trained pipeline:

```python
sample_predictions = trained_scorecard_model.predict(sample_customers)
sample_probabilities = trained_scorecard_model.predict_proba(sample_customers)[:, 1]
```

## Displayed decision rules

| Result | Displayed status |
| --- | --- |
| Predicted class is `0` | Application Rejected |
| Predicted class is `1`, probability at least 80% | Application Approved |
| Predicted class is `1`, probability from 20% to below 80% | Manual Review Pending |
| Predicted class is `1`, probability below 20% | Application Rejected |

These thresholds are demonstration policy rules, not values learned by the model. They require formal business, risk, fairness, and compliance approval before real use.

## Example session

Example inputs for a comparatively strong application:

```text
Enter the FICO Score:: 780
Enter the Annual Income:: 95000
Enter the Loan Amount:: 12000
Enter the Loan Term Moths:: 36
Enter the Employment Status Enter 1. Salaried, other number for Self Employed  :: 1
Enter the Years Employed:: 8
Enter the Savings Balance:: 25000
```

The exact result depends on the source data and generated training split. After a result, enter uppercase `Y` to score another applicant; any other response exits.

## Public interfaces

### `read_float`

```python
success, value = read_float(prompt)
```

Reads one numeric response. It returns a Boolean success flag and a NumPy integer. Invalid input returns `(False, 0)` after printing an error.

### `validate_split_data`

```python
validate_split_data(
    train_features,
    test_features,
    train_target,
    test_target,
)
```

Rejects empty features, mismatched feature/target lengths, and missing input columns. It returns `None` when validation succeeds.

### `create_preprocessor`

```python
preprocessor = create_preprocessor(
    scale_numeric=True,
    dense_output=False,
)
```

Returns an unfitted `ColumnTransformer`. `scale_numeric` controls standardization. `dense_output` controls the one-hot encoder output format.

### `evaluate_model`

```python
roc_auc, accuracy, matrix, report = evaluate_model(
    model,
    test_features,
    test_target,
)
```

Evaluates a fitted binary classifier using test predictions and positive-class probabilities.

### `credit_scorecard.get_model`

```python
scorecard = credit_scorecard()
model, accuracy = scorecard.get_model(
    train_features=train_features,
    test_features=test_features,
    train_target=train_target,
    test_target=test_target,
)
```

Returns a fitted preprocessing/logistic-regression pipeline and test accuracy. It also prints the other evaluation metrics.

### `random_forest.get_model`

```python
forest = random_forest()
model, matrix, accuracy, report = forest.get_model(
    train_features=train_features,
    test_features=test_features,
    train_target=train_target,
    test_target=test_target,
)
```

Returns a fitted preprocessing/random-forest pipeline, confusion matrix, accuracy, and classification report.

## Reusing the module

Importing the module exposes its reusable interfaces without starting the prompt:

```python
from loan_app import credit_scorecard, create_preprocessor
```

Callers are responsible for loading, cleaning, validating, and splitting data before invoking a model's `get_model` method.

## Error behavior

- Missing CSV: pandas raises `FileNotFoundError`.
- Missing columns: the application raises `ValueError` with column names.
- Failed quality checks: one `ValueError` lists all violations.
- Invalid interactive value: `read_float` prints an error and returns failure.
- A test target with only one class can prevent ROC-AUC calculation.

## Current limitations

- `read_float` accepts decimal text but converts it to `np.int64`, truncating fractions.
- The program trains a new model every time it starts and does not save it.
- New applications can be entered only through standard input.
- The validation split is not used for tuning or threshold selection.
- Probability is calculated in the interactive flow only for positive predictions.
- Missing FICO values are represented by `0`, which can affect model behavior.
- Historical approval is not a repayment-performance outcome.
- Predictions and probabilities are not adverse-action explanations.
- There is no authentication, persistence, audit log, API, or production monitoring.

## Production safeguards

- Do not use `applicant_id` as a model feature.
- Replace historical approval with a properly defined outcome target.
- Validate calibration and performance on independent, time-separated data.
- Evaluate group-level outcomes and proxy variables under fair-lending requirements.
- Govern thresholds outside the estimator implementation.
- Version data, preprocessing, parameters, thresholds, and decisions.
- Monitor drift, calibration, overrides, appeals, and operational incidents.


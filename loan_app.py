
"""Train and run an interactive loan-approval classification application.

The module loads historical applications from ``data/loan_applications.csv``,
cleans and validates the data, creates stratified train/validation/test splits,
and trains a logistic-regression credit-scorecard pipeline. When executed as a
script, it prompts for a new applicant's details and reports an approval,
manual-review, or rejection result based on the trained model.

The target is the historical ``approved`` field. Consequently, predictions
represent patterns in prior approval decisions rather than validated estimates
of repayment or default. The application is a demonstration and requires legal,
fair-lending, calibration, governance, and independent validation work before
any production lending use.

Typical usage::

    python loan_app.py

The reusable interfaces, feature schema, metrics, and operational limitations
are documented in ``LOAN_APP_INTERFACE.MD``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path("data") / "loan_applications.csv"
ID_COLUMN = "applicant_id"
TARGET_COLUMN = "approved"
POSITIVE_CLASS = "Approved"

# Prompts are ordered to match INPUT_COLUMNS. The fifth response is interpreted
# as an employment-status selection rather than a continuous numeric feature.
DATA_READ_MESSAGES = [
    'Enter the FICO Score:',
    'Enter the Annual Income:',
    'Enter the Loan Amount:',
    'Enter the Loan Term Moths:',
    'Enter the Employment Status Enter 1. Salaried, other number for Self Employed  :',
    'Enter the Years Employed:',
    'Enter the Savings Balance:'        
]

SELF_EMPLOYED = "Self-employed"
SALARIED = "Salaried"

RANDOM_STATE = 42
# Ordered schema required by every fitted model pipeline and prediction request.
INPUT_COLUMNS = [
    "fico_score",
    "annual_income",
    "loan_amount",
    "loan_term_months",
    "employment_status",
    "years_employed",
    "savings_balance",
]

NUMERIC_FEATURES = (
    "fico_score",
    "annual_income",
    "loan_amount",
    "loan_term_months",
    "years_employed",
    "savings_balance",
)

CATEGORICAL_FEATURES = ("employment_status",)

RANDOM_FOREST_PARAMS = {
    "class_weight": "balanced",
    "min_samples_leaf": 5,
    "n_estimators": 300,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

LOGISTIC_REGRESSION_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1_000,
    "random_state": RANDOM_STATE,
}

MINIMUM_TRAINING_ROWS = 100
MINIMUM_CLASS_COUNT = 20
MAXIMUM_MISSING_RATE = 0.20
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
CROSS_VALIDATION_FOLDS = 5
CALIBRATION_METHOD = "sigmoid"
MINIMUM_TRAINING_ROWS = 100
MINIMUM_CLASS_COUNT = 20
MAXIMUM_MISSING_RATE = 0.20

fico_score = np.int64(0)
annual_income = np.int64(0)
loan_amount = np.int64(0)
loan_term_months = np.int64(0)
employment_status = SELF_EMPLOYED
years_employed = np.int64(0)
savings_balance = np.int64(0)

def evaluate_model(
    model: Pipeline,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> tuple[float, float, np.ndarray, str]:
    """Evaluate a fitted binary-classification pipeline on held-out data.

    Args:
        model: Fitted preprocessing and classifier pipeline.
        test_features: Held-out features containing every ``INPUT_COLUMNS`` field.
        test_target: Binary target values aligned with ``test_features``.

    Returns:
        A tuple containing ROC-AUC, accuracy, the confusion-matrix array, and a
        text classification report, in that order.

    Raises:
        ValueError: If ROC-AUC cannot be calculated, such as when the test target
            contains only one class.
    """
    test_inputs = test_features.loc[:, INPUT_COLUMNS]
    test_predictions = model.predict(test_inputs)
    test_probabilities = model.predict_proba(test_inputs)[:, 1]
    return (
        float(roc_auc_score(test_target, test_probabilities)),
        float(accuracy_score(test_target, test_predictions)),
        confusion_matrix(test_target, test_predictions),
        classification_report(test_target, test_predictions, zero_division=0),
    )

def create_preprocessor(
    *,
    scale_numeric: bool,
    dense_output: bool,
) -> ColumnTransformer:
    """Build preprocessing shared by training and inference.

    Numeric columns receive median imputation and optional standardization.
    Categorical columns receive most-frequent imputation and one-hot encoding.

    Args:
        scale_numeric: Whether to standardize numeric values after imputation.
        dense_output: Whether one-hot encoding should return a dense array.

    Returns:
        An unfitted column transformer suitable for a scikit-learn pipeline.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_preprocessor = Pipeline(steps=numeric_steps)
    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense_output,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessor, list(NUMERIC_FEATURES)),
            ("categorical", categorical_preprocessor, list(CATEGORICAL_FEATURES)),
        ]
    )

def validate_split_data(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
) -> None:
    """Validate the minimum structural contract for model inputs.

    Args:
        train_features: Feature rows used to fit the pipeline.
        test_features: Held-out feature rows used to evaluate the pipeline.
        train_target: Target values aligned with the training rows.
        test_target: Target values aligned with the test rows.

    Raises:
        ValueError: If either feature set is empty, feature and target lengths do
            not match, or a required input column is absent.
    """
    if train_features.empty or test_features.empty:
        raise ValueError("Training and test features cannot be empty.")
    if len(train_features) != len(train_target):
        raise ValueError("Training features and target must have equal lengths.")
    if len(test_features) != len(test_target):
        raise ValueError("Test features and target must have equal lengths.")

    missing_columns = set(INPUT_COLUMNS) - set(train_features.columns)
    missing_columns |= set(INPUT_COLUMNS) - set(test_features.columns)
    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Model input columns are missing: {missing_names}.")


class random_forest:
    """Build and evaluate a random-forest approval classifier pipeline."""

    def get_model(
        self,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        train_target: pd.Series,
        test_target: pd.Series,
    ) -> tuple[Pipeline, np.ndarray, float, str]:
        """Train a random forest and evaluate it on the supplied test split.

        Args:
            train_features: Raw training features in the application schema.
            test_features: Raw held-out features in the application schema.
            train_target: Binary outcomes aligned with ``train_features``.
            test_target: Binary outcomes aligned with ``test_features``.

        Returns:
            The fitted pipeline, confusion matrix, accuracy, and classification
            report, in that order.

        Raises:
            ValueError: If the split data fails structural validation.
        """
        validate_split_data(
            train_features,
            test_features,
            train_target,
            test_target,
        )

        preprocessor = create_preprocessor(
            scale_numeric=False,
            dense_output=False,
        )
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(**RANDOM_FOREST_PARAMS),
                ),
            ]
        )

        model.fit(train_features.loc[:, INPUT_COLUMNS], train_target)
        (
            _,
            model_accuracy,
            model_confusion_matrix,
            model_classification_report,
        ) = evaluate_model(
            model,
            test_features,
            test_target,
        )

        return (
            model,
            model_confusion_matrix,
            model_accuracy,
            model_classification_report,
        )
    
def read_float(message_p: str) -> tuple[bool, np.int64 | int]:
    """Prompt for a numeric value and convert it to a NumPy integer.

    Args:
        message_p: Prompt displayed before reading standard input.

    Returns:
        A pair containing a success flag and the converted integer. When input
        is invalid, the function prints an error and returns ``(False, 0)``.
    """
    user_input = input((message_p +":"))
    try:
        float_value = float(user_input)
        print(f"Read float from file: {float_value}")
        return True, np.int64(float_value)
    except ValueError:
        print("The file did not contain a valid float.")
        return False, 0

class credit_scorecard:
    """Build and evaluate the logistic-regression credit-scorecard pipeline."""

    def get_model(
        self,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        train_target: pd.Series,
        test_target: pd.Series,
    ) -> tuple[Pipeline, float]:
        """Train the scorecard pipeline and evaluate held-out applications.

        Args:
            train_features: Raw training features in the application schema.
            test_features: Raw held-out features in the application schema.
            train_target: Binary outcomes aligned with ``train_features``.
            test_target: Binary outcomes aligned with ``test_features``.

        Returns:
            The fitted preprocessing/classifier pipeline and its test accuracy.
            ROC-AUC, confusion matrix, and the classification report are printed
            as diagnostic output.

        Raises:
            ValueError: If the split data fails structural validation.
        """
        validate_split_data(
            train_features,
            test_features,
            train_target,
            test_target,
        )

        preprocessor = create_preprocessor(
            scale_numeric=True,
            dense_output=False,
        )
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
                ),
            ]
        )

        model.fit(train_features.loc[:, INPUT_COLUMNS], train_target)
        (
            model_roc_auc,
            model_accuracy,
            model_confusion_matrix,
            model_classification_report,
        ) = evaluate_model(
            model,
            test_features,
            test_target,
        )

        print(f"Accuracy score: {model_accuracy:.4f}")
        print(f"ROC-AUC score: {model_roc_auc:.4f}")
        print("Confusion matrix:")
        print(model_confusion_matrix)
        print("Classification report:")
        print(model_classification_report)

        return model, model_accuracy


if __name__ == "__main__":
    # Stage 1: Load source data and normalize currency and FICO fields.
    print('START')
    loan_applications = pd.read_csv(DATA_PATH)
    print("Schema before cleaning:")
    print(loan_applications.dtypes)
    currency_columns = ["annual_income", "loan_amount", "savings_balance"]
    loan_applications[currency_columns] = loan_applications[currency_columns].apply(
        lambda column: pd.to_numeric(
                column.astype("string").str.replace(r"[$,']", "", regex=True),
                errors="coerce",
            )
        )

    loan_applications = loan_applications.loc[
        ~loan_applications["annual_income"].lt(0)
    ].copy()

    loan_applications["fico_score"] = (
        pd.to_numeric(loan_applications["fico_score"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )

    loan_applications = loan_applications.drop_duplicates(
        subset=ID_COLUMN,
        keep="first",
    ).copy()

    print("\nSchema after cleaning:")
    print(loan_applications.dtypes)

    print(loan_applications)

    # Stage 2: Enforce dataset schema and minimum quality requirements before
    # fitting any preprocessing or classifier state.
    required_columns = {ID_COLUMN, TARGET_COLUMN, *INPUT_COLUMNS}
    missing_columns = required_columns - set(loan_applications.columns)
    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Required columns are missing: {missing_names}.")

    row_count = len(loan_applications)
    duplicate_applicant_count = int(loan_applications[ID_COLUMN].duplicated().sum())
    missing_rates = loan_applications.loc[:, INPUT_COLUMNS].isna().mean()
    target_class_counts = loan_applications[TARGET_COLUMN].value_counts(dropna=False)
    invalid_numeric_counts = {
        "annual_income": int((loan_applications["annual_income"] < 0).sum()),
        "loan_amount": int((loan_applications["loan_amount"] <= 0).sum()),
        "loan_term_months": int((loan_applications["loan_term_months"] <= 0).sum()),
        "years_employed": int((loan_applications["years_employed"] < 0).sum()),
        "savings_balance": int((loan_applications["savings_balance"] < 0).sum()),
    }

    quality_violations = []
    if row_count < MINIMUM_TRAINING_ROWS:
        quality_violations.append(
            f"Only {row_count} rows are available; at least {MINIMUM_TRAINING_ROWS} are required."
        )
    if duplicate_applicant_count:
        quality_violations.append(
            f"Found {duplicate_applicant_count} duplicate applicant IDs."
        )
    if missing_rates.max() > MAXIMUM_MISSING_RATE:
        quality_violations.append(
            f"At least one feature exceeds the {MAXIMUM_MISSING_RATE:.0%} missing-value limit."
        )
    if loan_applications[TARGET_COLUMN].isna().any():
        quality_violations.append("The target column contains missing values.")
    if target_class_counts.min() < MINIMUM_CLASS_COUNT:
        quality_violations.append(
            f"At least one target class has fewer than {MINIMUM_CLASS_COUNT} rows."
        )
    if any(invalid_numeric_counts.values()):
        quality_violations.append("One or more numeric features contain invalid values.")

    data_quality_report = {
        "row_count": row_count,
        "duplicate_applicant_count": duplicate_applicant_count,
        "missing_rates": missing_rates.to_dict(),
        "target_class_counts": target_class_counts.to_dict(),
        "invalid_numeric_counts": invalid_numeric_counts,
        "passed": not quality_violations,
    }

    if quality_violations:
        raise ValueError("Data quality checks failed: " + " ".join(quality_violations))

    print(data_quality_report)


    # Stage 3: Encode the historical approval target and create reproducible,
    # stratified train/validation/test partitions.
    input_features = loan_applications.loc[:, INPUT_COLUMNS].copy()
    target = loan_applications[TARGET_COLUMN].eq(POSITIVE_CLASS).astype("int8")

    train_validation_features, test_features, train_validation_target, test_target = (
        train_test_split(
            input_features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=target,
        )
    )

    validation_fraction_of_remaining = VALIDATION_SIZE / (1 - TEST_SIZE)
    train_features, validation_features, train_target, validation_target = (
        train_test_split(
            train_validation_features,
            train_validation_target,
            test_size=validation_fraction_of_remaining,
            random_state=RANDOM_STATE,
            stratify=train_validation_target,
        )
    )

    dataset_split_summary = pd.DataFrame(
        {
            "rows": [len(train_features), len(validation_features), len(test_features)],
            "positive_rate": [
                train_target.mean(),
                validation_target.mean(),
                test_target.mean(),
            ],
        },
        index=["train", "validation", "test"],
    )

    print('dataset_split_summary:', dataset_split_summary)


    # Stage 4: Fit and evaluate the credit-scorecard pipeline. All preprocessing
    # is contained in the returned model and is therefore reused for inference.
    credit_scorecard_instance = credit_scorecard()
    trained_scorecard_model, scorecard_accuracy = (
        credit_scorecard_instance.get_model(
            train_features=train_features,
            test_features=test_features,
            train_target=train_target,
            test_target=test_target,
        )
    )
    print(f"Credit scorecard accuracy score: {scorecard_accuracy:.4f}")

    '''scorecard_predictions = trained_scorecard_model.predict(sample_customers)
    scorecard_probabilities = trained_scorecard_model.predict_proba(
        sample_customers
    )[:, 1]

    minimum_score_points = 300
    maximum_score_points = 850
    scorecard_points = np.rint(
        minimum_score_points
        + scorecard_probabilities * (maximum_score_points - minimum_score_points)
    ).astype("int64")

    scorecard_results = pd.DataFrame(
        {
            "prediction": np.where(
                scorecard_predictions == 1,
                POSITIVE_CLASS,
                "Not Approved",
            ),
            "approval_probability": scorecard_probabilities,
            "score_points": scorecard_points,
        },
        index=sample_customers.index,
    )

    print("\nCredit scorecard sample predictions and points:")
    print(scorecard_results)'''


    # Stage 5: Collect new applications interactively and score them with the
    # fitted pipeline. The thresholds below are demonstration policy rules.
    print('***************************************')
    c_flag = True
    while c_flag:
        print('New customer Input:')
        index = 0
        for msg in DATA_READ_MESSAGES:
            flag_read, value = read_float(msg)
            index +=1
            #print('index:', index)
            if flag_read:
                match index:
                    case 1:
                        fico_score = value
                    case 2:
                        annual_income = value
                    case 3:
                        loan_amount = value
                    case 4:
                        loan_term_months = value
                    case 5:
                        if value == 1:
                            employment_status = SALARIED
                        else:
                            employment_status = SELF_EMPLOYED
                    case 6:
                        years_employed = value
                    case 7:
                        savings_balance = value
                    case _:
                        print('No Action needed')
            else:
                break

            if index == 7:
                #TODO do the prediction ..
                print('fico_score:', fico_score)
                print('annual_income:', annual_income)
                print('loan_amount:', loan_amount)
                print('loan_term_months:', loan_term_months)
                print('employment_status:', employment_status)
                print('years_employed:', years_employed)
                print('savings_balance:', savings_balance)

                sample_customers = pd.DataFrame(
                    [
                        {
                            "fico_score": fico_score,
                            "annual_income": annual_income,
                            "loan_amount": loan_amount,
                            "loan_term_months": loan_term_months,
                            "employment_status": employment_status,
                            "years_employed": years_employed,
                            "savings_balance": savings_balance,
                        }
                    ]
                )

                # Pass the original feature schema to the pipeline; it applies
                # its fitted imputing, scaling, and encoding transformations.
                sample_predictions = trained_scorecard_model.predict(sample_customers)
                print('****** Application Status - PROCESSING - START ******')
                print('Prediction Score:', sample_predictions[0])
                if sample_predictions[0] == 1:
                    sample_probabilities = trained_scorecard_model.predict_proba(sample_customers)[:, 1]
                    prab_percentage = sample_probabilities[0] * 100
                    print('prab_percentage:', prab_percentage)
                    if prab_percentage >= 80:
                        print('Application Approved')
                    elif prab_percentage >= 20:    
                        print('Application Received. Manual Review Pending.')
                    else:
                        print('Application Rejected')    
                else:
                    print('Application Rejected')
                print('****** Application Status  - PROCESSING - END ******')

        user_input_continue = input("Enter your options ... Continue Y, Quit any other key: ")
        if user_input_continue.strip() == 'Y':
            c_flag = True
        else:
            c_flag = False

    print('END')

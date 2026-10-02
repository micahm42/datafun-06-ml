"""
src/datafun/home_sales.py - Home Sales Linear Regression Project.

Predict home price using square footage.
"""

# === DECLARE IMPORTS ===

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header, log_path
import matplotlib.pyplot as plt
from ml_vizkit import save_chart
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# === CONFIGURE LOGGER ===

LOG: logging.Logger = get_logger("P06-HOME-SALES", level="DEBUG")


# === LOCATE THE DATA FILE ===

DATA_FILE_PATH: Final[Path] = Path("data") / "home_sales.csv"


# === LOCATE THE CHART OUTPUT ===

CHART_DIR: Final[Path] = Path("docs") / "images"

PREDICTION_CHART_PATH: Final[Path] = CHART_DIR / "home-sales-regression-predictions.png"

RESIDUAL_CHART_PATH: Final[Path] = CHART_DIR / "home-sales-regression-residuals.png"


# === DETERMINE WHAT ONE ROW REPRESENTS ===

GRAIN: Final[str] = "one home sale"


# === DECLARE THE TARGET ===

TARGET_COLUMN: Final[str] = "price"


# === DECLARE THE FEATURE ===

FEATURE_COLUMN: Final[str] = "sqft"


# === DOCUMENT WHY THE FEATURE MIGHT HELP ===

FEATURE_DECISION: Final[str] = r"""
I want to predict home sale price.

I selected square footage as the feature.

Square footage provides a measurement of the size
of the home. Larger homes may have higher sale prices
because they provide more living space.

Square footage might therefore contain useful information
for predicting home prices.

I do not know yet how well square footage will predict
home prices. The modeling process will provide evidence.
"""


# === DECLARE THE TRAIN / TEST SPLIT ===

TEST_FRACTION: Final[float] = 0.20

RANDOM_SEED: Final[int] = 42


# === DOCUMENT THE SPLIT DECISION ===

SPLIT_DECISION: Final[str] = r"""
I will use 80% of the modeling rows for training
and hold back 20% for testing.

I want most of the available data to be available
for learning the model while keeping a separate
set of observations that the model did not see
during training.

The test rows will be used to evaluate how the
trained model performs on unseen observations.

I will use a random seed of 42 so the split is
reproducible.
"""


# === DECLARE THE BASELINE ===

BASELINE_STRATEGY: Final[str] = "mean"


# === DOCUMENT THE BASELINE DECISION ===

BASELINE_DECISION: Final[str] = r"""
Before evaluating the LinearRegression model,
I need a simple baseline for comparison.

The baseline will ignore square footage and predict
the average home price from the training data for
every test observation.

A useful predictive model should improve on this
simple reference prediction.
"""


# === DOCUMENT THE MODEL DECISION ===

MODEL_DECISION: Final[str] = r"""
I will use LinearRegression.

Linear regression fits a straight-line relationship
between square footage and home price.

This provides a simple and interpretable model
that can be compared with the baseline.

Fitting a line does not prove that a straight-line
relationship is a good description of home prices.

The evaluation metrics and residual plot will help
assess whether the model is useful.
"""


# === DEFINE THE MAIN FUNCTION ===


def main() -> None:
    """Run the home sales linear regression analysis."""

    log_header(LOG, "P06 - LINEAR REGRESSION - HOME SALES")

    LOG.info("===================================")
    LOG.info("START main()")
    LOG.info("===================================")

    # === 01. OBSERVE ===

    LOG.info("-------------------------------")
    LOG.info("01. OBSERVE the data.")
    LOG.info("-------------------------------")

    log_path(LOG, "data file", path=DATA_FILE_PATH)

    df: pd.DataFrame = pd.read_csv(DATA_FILE_PATH)

    LOG.info("Data loaded successfully.")
    LOG.info(f"Grain: {GRAIN}")
    LOG.info(f"Rows: {df.shape[0]}")
    LOG.info(f"Columns: {df.shape[1]}")
    LOG.info(f"Column names: {df.columns.tolist()}")

    # === 02. DECLARE TARGET AND FEATURE ===

    LOG.info("-------------------------------")
    LOG.info("02. DECLARE the target and feature.")
    LOG.info("-------------------------------")

    LOG.info(f"Target:  {TARGET_COLUMN}")
    LOG.info(f"Feature: {FEATURE_COLUMN}")
    LOG.info(FEATURE_DECISION)

    # === 03. PREPARE MODELING DATA ===

    LOG.info("-------------------------------")
    LOG.info("03. PREPARE the modeling data.")
    LOG.info("-------------------------------")

    required_columns: list[str] = [
        FEATURE_COLUMN,
        TARGET_COLUMN,
    ]

    df_model: pd.DataFrame = df.dropna(subset=required_columns).copy()

    count_original: int = df.shape[0]
    count_model: int = df_model.shape[0]
    count_dropped: int = count_original - count_model

    LOG.info(f"Original rows: {count_original}")
    LOG.info(f"Modeling rows: {count_model}")
    LOG.info(f"Rows dropped: {count_dropped}")

    X: pd.DataFrame = df_model[[FEATURE_COLUMN]]
    y: pd.Series = df_model[TARGET_COLUMN]

    LOG.info(f"X shape: {X.shape}")
    LOG.info(f"y shape: {y.shape}")

    # === 04. SPLIT DATA ===

    LOG.info("-------------------------------")
    LOG.info("04. SPLIT into training and test data.")
    LOG.info("-------------------------------")

    LOG.info(SPLIT_DECISION)

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
    )

    LOG.info(f"Training rows: {X_train.shape[0]}")
    LOG.info(f"Test rows: {X_test.shape[0]}")

    # === 05. BASELINE ===

    LOG.info("-------------------------------")
    LOG.info("05. BASELINE with a simple reference model.")
    LOG.info("-------------------------------")

    LOG.info(BASELINE_DECISION)

    baseline_model = DummyRegressor(
        strategy=BASELINE_STRATEGY,
    )

    baseline_model.fit(
        X_train,
        y_train,
    )

    baseline_predictions: np.ndarray = baseline_model.predict(X_test)

    baseline_rmse: float = float(
        root_mean_squared_error(
            y_test,
            baseline_predictions,
        )
    )

    baseline_r_squared: float = float(
        r2_score(
            y_test,
            baseline_predictions,
        )
    )

    LOG.info(f"Baseline strategy: {BASELINE_STRATEGY}")
    LOG.info(f"Baseline RMSE: {baseline_rmse:.2f}")
    LOG.info(f"Baseline R-squared: {baseline_r_squared:.3f}")

    # === 06. TRAIN MODEL ===

    LOG.info("-------------------------------")
    LOG.info("06. TRAIN a LinearRegression model.")
    LOG.info("-------------------------------")

    LOG.info(MODEL_DECISION)

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    slope: float = float(model.coef_[0])
    intercept: float = float(model.intercept_)

    LOG.info("The model learned this line:")
    LOG.info(f"{TARGET_COLUMN} = {slope:.6g} * {FEATURE_COLUMN} + {intercept:.6g}")

    # === 07. PREDICT ===

    LOG.info("-------------------------------")
    LOG.info("07. PREDICT on X_test.")
    LOG.info("-------------------------------")

    model_predictions: np.ndarray = model.predict(X_test)

    LOG.info(f"Predictions created: {len(model_predictions)}")

    # === 08. EVALUATE ===

    LOG.info("-------------------------------")
    LOG.info("08. EVALUATE baseline vs model on y_test.")
    LOG.info("-------------------------------")

    model_rmse: float = float(
        root_mean_squared_error(
            y_test,
            model_predictions,
        )
    )

    model_r_squared: float = float(
        r2_score(
            y_test,
            model_predictions,
        )
    )

    LOG.info("BASELINE RESULTS")
    LOG.info(f"RMSE:      {baseline_rmse:.2f}")
    LOG.info(f"R-squared: {baseline_r_squared:.3f}")

    LOG.info("LINEAR REGRESSION RESULTS")
    LOG.info(f"RMSE:      {model_rmse:.2f}")
    LOG.info(f"R-squared: {model_r_squared:.3f}")

    # === 09. VISUALIZE ===

    LOG.info("-------------------------------")
    LOG.info("09. VISUALIZE predictions and residuals.")
    LOG.info("-------------------------------")

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # === PREDICTION CHART ===

    _prediction_figure, prediction_ax = plt.subplots()

    x_test_values: np.ndarray = X_test[FEATURE_COLUMN].to_numpy()

    y_test_values: np.ndarray = y_test.to_numpy()

    prediction_ax.scatter(
        x_test_values,
        y_test_values,
        label="Actual",
    )

    prediction_order: np.ndarray = np.argsort(x_test_values)

    prediction_ax.plot(
        x_test_values[prediction_order],
        model_predictions[prediction_order],
        label="Predicted",
    )

    prediction_ax.set_title("Square Footage vs. Home Price")

    prediction_ax.set_xlabel("Square Footage")

    prediction_ax.set_ylabel("Home Price")

    prediction_ax.legend()

    save_chart(
        prediction_ax,
        PREDICTION_CHART_PATH,
    )

    LOG.info(f"Chart saved successfully at {PREDICTION_CHART_PATH}.")

    # === RESIDUAL CHART ===

    residuals: np.ndarray = y_test_values - model_predictions

    _residual_figure, residual_ax = plt.subplots()

    residual_ax.scatter(
        x_test_values,
        residuals,
    )

    residual_ax.axhline(0)

    residual_ax.set_title("Residuals for Home Price Model")

    residual_ax.set_xlabel("Square Footage")

    residual_ax.set_ylabel("Residual (Actual - Predicted Price)")

    save_chart(
        residual_ax,
        RESIDUAL_CHART_PATH,
    )

    LOG.info(f"Chart saved successfully at {RESIDUAL_CHART_PATH}.")

    # === 10. ASSESS RESULTS ===

    LOG.info("-------------------------------")
    LOG.info("10. ASSESS the results.")
    LOG.info("-------------------------------")

    LOG.info(r"""CUSTOM OBSERVATIONS:

I used square footage to predict home prices.

The baseline RMSE was ...

The LinearRegression RMSE was ...

Compared with the baseline,
the LinearRegression model ...

The model R-squared was ...

In the residual plot, I observed ...

Based on this evidence,
I conclude ...

Next, I would like to try using another
feature such as bedrooms or bathrooms.
""")

    LOG.info("===================================")
    LOG.info("END main() - Executed successfully!")
    LOG.info("===================================")

    plt.show()


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()

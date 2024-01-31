import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from steps.config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    config: ModelNameConfig
) -> RegressorMixin:
    """
    Trains the model on the ingested data.

    Args:
        X_train: pd.DataFrame
        y_train: pd.DataFrame
        config: ModelNameConfig

    Returns:
        RegressorMixin: Trained regression model.
    """

    # Log the input arguments
    logging.info(f"train_model received X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, config: {config}")

    # Instantiate a new model and fit it to the provided data
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            model.train(X_train, y_train)
            return model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e

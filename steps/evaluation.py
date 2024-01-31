import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import MSE, RMSE, R2
from typing import Tuple
from typing_extensions import Annotated
from .model_train import train_model  # Import the train_model step
from .config import ModelNameConfig

@step
def evaluate_model(model: RegressorMixin,
                   X_train: pd.DataFrame,
                   y_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,
                   config: ModelNameConfig) -> Tuple[
                       Annotated[float, "r2_score"],
                       Annotated[float, "rmse"],
                       Annotated[float, "mse_score"]
                   ]:
    """
    Evaluate the model on the ingested data.

    Args:
        model: RegressorMixin
        X_train (pd.DataFrame): The training features.
        y_train (pd.DataFrame): The training labels.
        X_test (pd.DataFrame): The test features.
        y_test (pd.DataFrame): The true labels.
        config: ModelNameConfig
    """
    # Logging example
    logging.info("Evaluating model with ingested data...")

    try:
        # Check if the model is fitted before making predictions
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            # Assuming train_model is the method to fit the model
            trained_model = train_model(X_train, y_train, config)
            model = trained_model  # Update the model with the trained one

        prediction = model.predict(X_test)

        mse_class = MSE()
        mse_score: float = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2_score: float = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse_score: float = rmse_class.calculate_scores(y_test, prediction)

        return r2_score, rmse_score, mse_score
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e
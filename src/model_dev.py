import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin

class Model(ABC):
    """Abstract class for all models"""

    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        pass

class LinearRegressionModel(LinearRegression, Model, RegressorMixin):
    """Linear Regression Class"""

    def __init__(self):
        self.is_fitted = False

    def train(self, X_train, y_train, **kwargs):
        try:
            # Use self.fit instead of creating a new LinearRegression instance
            self.fit(X_train, y_train, **kwargs)
            self.is_fitted = True
            logging.info("Model training completed")
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call 'train' before using 'predict'.")
        return self.predict(X)


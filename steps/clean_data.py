import logging
import pandas as pd
from zenml import step
from typing import Tuple
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataDivideStrategy

@step
def clean_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Cleans the Data and returns the cleaned DataFrame.

    Args:
        df: Raw data

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Cleaned DataFrame.
    """
    try:
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()

        logging.info("Data cleaning completed")
        # Return the cleaned DataFrame as a tuple
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e

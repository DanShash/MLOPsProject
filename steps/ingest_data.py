import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting Data from the data_path
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): Path to the data
        """
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        """
        Ingesting the data from the data_path.
        Returns:
            pd.DataFrame: The ingested data.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path.

    Args:
        data_path (str): Path to the data.
    
    Returns:
        pd.DataFrame: The ingested data.
    """
    my_data = IngestData(data_path)
    ingested_data = my_data.get_data()
    return ingested_data


    
    
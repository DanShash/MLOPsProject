from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig 

@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    # Ingest and clean data
    df = ingest_df(data_path)
    cleaned_data = clean_df(df)
    
    # Unpack cleaned data
    X_train, X_test, y_train, y_test = cleaned_data
    
    # Get the model configuration
    config = ModelNameConfig()
    
    # Train the model
    model = train_model(X_train, y_train, config)
    
    # Evaluate the model
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    
    # Optionally, you can retrain and re-evaluate using the original dataframe
    # train_model(df)
    # evaluate_model(df)
    
    # Return any results or data if needed
    return r2_score, rmse


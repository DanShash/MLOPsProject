import pandas as pd

def check_columns_exist(csv_file_path, columns_to_check):
    # Read only the header of the CSV file
    header = pd.read_csv(csv_file_path, nrows=0).columns

    # Check if all columns_to_check exist in the header
    # Removing leading and trailing whitespaces
    header = [col.strip() for col in header]
    
    missing_columns = [col for col in columns_to_check if col not in header]

    if missing_columns:
        print(f"The following columns are missing: {missing_columns}")
    else:
        print("All columns exist in the CSV file.")


csv_file_path = "C:\Users\???????\Desktop\MLOPs Project\data\olist_customers_dataset.csv"
columns_to_check = [
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date',
    'order_purchase_timestamp'
]

check_columns_exist(csv_file_path, columns_to_check)

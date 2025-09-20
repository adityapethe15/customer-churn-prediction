import os
from .data_preprocessing import load_data, preprocess_data, encode_features
from .model_training import train_and_evaluate_models, get_feature_importances

def main():
    # Define file path
    data_path = 'data/Telco-Customer-Churn.csv' # or 'data/Telco-Customer-Churn.xlsx'

    # Check if the data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at '{data_path}'")
        print("Please download the Telco Customer Churn dataset from Kaggle and place it in the 'data' folder.")
        return

    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    df = load_data(data_path)
    df_processed = preprocess_data(df)
    
    # Step 2: Encode features
    print("Step 2: Encoding features...")
    X_encoded, y = encode_features(df_processed)
    
    # Step 3: Train and evaluate models
    print("Step 3: Training and evaluating models...")
    rf_model = train_and_evaluate_models(X_encoded, y)
    
    # Step 4: Get and print feature importances
    print("\nStep 4: Analyzing feature importances...")
    get_feature_importances(rf_model, X_encoded.columns)
    
    print("\nProject pipeline complete.")

if __name__ == '__main__':
    main()
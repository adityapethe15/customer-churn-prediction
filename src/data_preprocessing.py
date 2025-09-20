import pandas as pd
import numpy as np

def load_data(filepath):
    """Loads the dataset from a given file path."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

def preprocess_data(df):
    """Performs data cleaning and preprocessing."""
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def encode_features(df):
    """Encodes categorical features and separates X and y."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X_encoded, y
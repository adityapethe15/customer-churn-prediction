import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_and_evaluate_models(X, y):
    """Trains and evaluates both Logistic Regression and Random Forest models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Logistic Regression
    print("--- Training Logistic Regression Model ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    log_reg = LogisticRegression(solver='liblinear')
    log_reg.fit(X_train_scaled, y_train)
    y_pred_log_reg = log_reg.predict(X_test_scaled)
    
    print("\n--- Logistic Regression Performance ---")
    print(classification_report(y_test, y_pred_log_reg))
    print("ROC AUC Score:", roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:, 1]))
    
    # Random Forest
    print("\n--- Training Random Forest Model ---")
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)
    
    print("\n--- Random Forest Performance ---")
    print(classification_report(y_test, y_pred_rf))
    print("ROC AUC Score:", roc_auc_score(y_test, random_forest.predict_proba(X_test)[:, 1]))
    
    return random_forest

def get_feature_importances(model, feature_names):
    """Prints top feature importances for a given model."""
    importances = model.feature_importances_
    feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importances_df.sort_values('importance', ascending=False, inplace=True)
    print("\n--- Top 10 Feature Importances (Random Forest) ---")
    print(feature_importances_df.head(10))
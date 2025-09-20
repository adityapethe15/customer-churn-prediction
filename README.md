# Customer Churn Prediction Project

This project implements an **end-to-end machine learning pipeline** to predict customer churn for a telecom company. The main goal is to identify customers at risk of leaving, understand the key factors driving churn, and provide actionable insights to improve customer retention strategies.

## Project Highlights

- **Problem Statement:** Predict which customers are likely to leave the telecom service, enabling proactive retention efforts.  
- **Data Preprocessing:** Handles missing values, encodes categorical variables, and scales numerical features for model readiness.  
- **Modeling:** Implements **Logistic Regression** and **Random Forest** classifiers to predict churn. Performance is evaluated using accuracy, classification reports, and **ROC AUC scores**.  
- **Feature Importance:** Identifies the most influential factors affecting churn, such as tenure, monthly charges, and contract type, to guide business decisions.  
- **Pipeline Design:** Modular code structure with reusable scripts (`data_preprocessing.py`, `model_training.py`, `main.py`) for maintainability and scalability.

## Project Structure

customer_churn_project/
├── data/                        # Contains the dataset (Telco-Customer-Churn.csv)
├── src/                         
│   ├── data_preprocessing.py     # Loads and cleans the data
│   ├── model_training.py         # Trains models and evaluates performance
│   └── main.py                   # Runs the full pipeline
├── .gitignore
├── README.md
└── requirements.txt              # Python dependencies

## Key Results

- **Model Performance:** Random Forest performed better than Logistic Regression, achieving higher ROC AUC and F1 scores.  
- **Top Features Driving Churn:**  
  1. Contract Type  
  2. Tenure  
  3. Monthly Charges  
  4. Payment Method  
- **Business Insights:** Customers with short tenure, month-to-month contracts, and higher monthly charges are more likely to churn, allowing targeted retention strategies.

## Technologies Used

- Python  
- Pandas & NumPy  
- Scikit-learn  

## Author

**Aditya Pethe**

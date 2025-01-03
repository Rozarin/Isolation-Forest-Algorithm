import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# This Function helps us generate a synthetic financial fraud dataset that displays random datasets everytime the program runs.
def generate_financial_fraud_dataset():
    np.random.seed(None)

    # This set the number of dataset samples to generate
    n_normal = 1000  # Normal transactions
    n_fraud = 500    # Fraudulent transactions

    # This generate normal transaction datasets, that have average amounts and frequencies.
    normal_data = {
        "TransactionID": np.arange(1, n_normal + 1),
        "TransactionAmount": np.random.normal(loc=100, scale=20, size=n_normal), 
        "TransactionType": np.random.choice(["Purchase", "Transfer", "Withdrawal"], n_normal),
        "TransactionFrequency": np.random.normal(loc=10, scale=2, size=n_normal), 
        "Label": 0,  # 0 indicates normal transaction
    }

    # This generate fraudulent transaction datasets, that have unusually high amounts and frequencies, and likely a fraud type transaction.
    fraud_data = {
        "TransactionID": np.arange(n_normal + 1, n_normal + n_fraud + 1),
        "TransactionAmount": np.random.uniform(low=1000, high=5000, size=n_fraud), 
        "TransactionType": np.random.choice(["Transfer", "Withdrawal"], n_fraud), 
        "TransactionFrequency": np.random.uniform(low=20, high=50, size=n_fraud),
        "Label": 1,  # 1 indicates fraudulent transaction
    }

    # This here combine normal and fraudulent transaction datasets into a single dataset using pd.concat()
    #pd.concat() is used to combine different datasets 
    data = pd.concat([pd.DataFrame(normal_data), pd.DataFrame(fraud_data)], ignore_index=True)

    # This shuffle the dataset for randomness display everytime the program runs.
    data = data.sample(frac=1, random_state=None).reset_index(drop=True)

    # This saves the dataset to a CSV file (with 500 fraudulent transactions)
    data.to_csv("financial_fraud_dataset.csv", index=False)
    print("Synthetic financial fraud dataset created and saved as 'financial_fraud_dataset.csv'.")

    return data

# This function applies the Isolation Forest Algorithm for fraud detection in the dataset
def apply_isolation_forest(data):
    # This here selects relevant features for anomaly detection like 'TransactionAmount' and 'TransactionFrequency'
    features = data[['TransactionAmount', 'TransactionFrequency']]

    # This here initialize the Isolation Forest model
    iso_forest = IsolationForest(contamination=0.05, random_state=42)

    # This here predict anomalies (fraudulent transactions)
    data['Predicted_Fraud'] = iso_forest.fit_predict(features)

    # This here is used to convert prediction labels like how the '-1' indicates fraud, and '1' indicates normal
    data['Predicted_Fraud'] = data['Predicted_Fraud'].apply(lambda x: 1 if x == -1 else 0)

    # This displays the dataset with predicted fraud labels
    print("Dataset with predicted fraud labels:")
    print(data.head())

# This here is our Main function
if __name__ == "__main__":
    # This part let us call and display the dataset function
    dataset = generate_financial_fraud_dataset()

    # This part here let us apply the Isolation Forest algorithm for fraud detection
    apply_isolation_forest(dataset)

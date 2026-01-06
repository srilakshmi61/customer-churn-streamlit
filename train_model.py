import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("churn.csv")   # 🔴 change file name if needed

# Drop ID
df = df.drop(['CustomerID', 'Last Interaction'], axis=1)

# Encode target
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# One-hot encoding
categorical_cols = [
    'Gender', 'Subscription Type', 'Contract Length',
    'Payment Delay', 'Support Calls'
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model + columns
with open("churn_model.pkl", "wb") as f:
    pickle.dump((model, X.columns), f)

print("✅ Model saved as churn_model.pkl")

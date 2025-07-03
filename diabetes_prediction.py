
# Diabetes Prediction System using Machine Learning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
dataset = pd.read_csv("diabetes.csv")

# Display first 5 rows
print(dataset.head())

# Separate features and target variable
X = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Model Prediction
y_pred = model.predict(X_test_scaled)

# Model Evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict on New Data
new_data = np.array([[5,116,74,0,0,25.6,0.201,30]])
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

if prediction[0] == 0:
    print("\nPrediction: No Diabetes Risk")
else:
    print("\nPrediction: At Risk of Diabetes")

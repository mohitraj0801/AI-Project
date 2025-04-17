import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("irrigation_data.csv")  # Ensure you save your data in this file format

# Encode categorical variables
label_encoders = {}
categorical_columns = ["Soil_Type", "Crop_Type", "Geographical_Location"]

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoders for later use

# Features and target
X = data[["Soil_Type", "Crop_Type", "Avg_Temperature", "Moisture_Level", "Geographical_Location"]]
y = data["Irrigation_Type"]

y_le = LabelEncoder()
y = y_le.fit_transform(y)
label_encoders["Irrigation_Type"] = y_le  # Save encoder for target variable

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "irrigation_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model training complete. Saved as irrigation_model.pkl")

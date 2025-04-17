# save_crop_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample training data (dummy data, you can replace with real)
data = pd.DataFrame({
    'Previous Crop': [1, 2, 3, 4, 5, 6, 7],
    'Soil Type': [1, 2, 3, 4, 1, 2, 3],
    'Moisture Level': [25, 35, 30, 40, 20, 32, 28],
    'Nitrogen (N)': [50, 60, 40, 70, 65, 55, 45],
    'Phosphorus (P)': [30, 40, 20, 35, 45, 30, 25],
    'Potassium (K)': [20, 30, 25, 40, 35, 30, 20],
    'Crop': [1, 2, 3, 4, 5, 6, 7]  # Labels: 1=Wheat, 2=Rice, etc.
})

X = data.drop(columns=['Crop'])
y = data['Crop']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'crop_rotation_recommendation_model.pkl')
print("Model saved as 'crop_rotation_recommendation_model.pkl'")

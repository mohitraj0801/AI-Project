from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# --- Load models and encoders ---
irrigation_model = joblib.load("irrigation_model.pkl")
irrigation_label_encoders = joblib.load("label_encoders.pkl")

crop_model = joblib.load('crop_rotation_recommendation_model.pkl')

# --- Mappings for crop recommendation ---
previous_crop_mapping = {
    'Groundnut': 1, 'Millets': 2, 'Wheat': 3, 'Maize': 4,
    'Cotton': 5, 'Sorghum': 6, 'Barley': 7
}

soil_type_mapping = {
    'Loamy': 1, 'Clayey': 2, 'Sandy': 3, 'Saline': 4
}

crop_mapping = {
    1: 'Wheat', 2: 'Rice', 3: 'Millets', 4: 'Cotton',
    5: 'Groundnut', 6: 'Maize', 7: 'Sorghum', 8: 'Barley',
}

# --- Route: Predict Irrigation Type ---
@app.route("/predict", methods=["POST"])
def predict_irrigation():
    try:
        data = request.get_json()
        print("Irrigation - Received Data:", data)

        location = data["Geographical_Location"]
        crop = data["Crop_Type"]
        soil = data["Soil_Type"]
        temperature = float(data["Avg_Temperature"])
        moisture = float(data["Moisture_Level"])

        try:
            soil_encoded = irrigation_label_encoders["Soil_Type"].transform([soil])[0]
            crop_encoded = irrigation_label_encoders["Crop_Type"].transform([crop])[0]
            location_encoded = irrigation_label_encoders["Geographical_Location"].transform([location])[0]
        except ValueError:
            return jsonify({"error": "Invalid categorical input values"}), 400

        input_features = np.array([[soil_encoded, crop_encoded, temperature, moisture, location_encoded]])
        prediction = irrigation_model.predict(input_features)
        predicted_irrigation = irrigation_label_encoders["Irrigation_Type"].inverse_transform(prediction)[0]

        return jsonify({"Predicted_Irrigation_Type": predicted_irrigation})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Route: Crop Recommendation ---
@app.route('/crop_recommendation', methods=['POST'])
def crop_recommendation():
    try:
        data = request.get_json()
        print("Crop Recommendation - Received data:", data)

        previous_crop = data.get('Previous Crop')
        soil_type = data.get('Soil Type')
        moisture_level = data.get('Moisture Level')
        nitrogen = data.get('Nitrogen (N)')
        phosphorus = data.get('Phosphorus (P)')
        potassium = data.get('Potassium (K)')

        input_data = pd.DataFrame([{
            "Previous Crop": previous_crop_mapping.get(previous_crop, -1),
            "Soil Type": soil_type_mapping.get(soil_type, -1),
            "Moisture Level": moisture_level,
            "Nitrogen (N)": nitrogen,
            "Phosphorus (P)": phosphorus,
            "Potassium (K)": potassium
        }])

        prediction = crop_model.predict(input_data)
        print("Crop Recommendation - Prediction:", prediction)

        recommended_crop = crop_mapping.get(prediction[0], 'No prediction available')
        return jsonify({'Recommended Crop': str(recommended_crop)})

    except Exception as e:
        return jsonify({'error': str(e)})

# --- Run the combined app ---
if __name__ == "__main__":
    app.run(debug=True)

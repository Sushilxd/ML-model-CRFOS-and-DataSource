from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import ast
import difflib

app = Flask(__name__)

# Load models and encoders for crop recommendation
crop_model = joblib.load("../models/crop_recommendation_model.pkl")
region_encoder_crop = joblib.load("../models/Region_encoder.pkl")
soil_encoder_crop = joblib.load("../models/Soil_Type_encoder.pkl")
moisture_encoder_crop = joblib.load("../models/moisture_encoder.pkl")

# Load models and encoders for fertilizer optimization
model_N = joblib.load("../models/fert_adj_model_N.pkl")
model_P = joblib.load("../models/fert_adj_model_P.pkl")
model_K = joblib.load("../models/fert_adj_model_K.pkl")
region_encoder_fert = joblib.load("../models/region_encoder_fert_adj.pkl")
soil_encoder_fert = joblib.load("../models/soil_encoder_fert_adj.pkl")
crop_encoder_fert = joblib.load("../models/crop_encoder_fert_adj.pkl")

# Cleaning function
def clean_text(val):
    return str(val).strip()

# Cleaning function for fertizlier
def clean_text_fertilizer(val):
    return str(val).strip().lower()

# Soil type matching function
def match_soil_type(user_input, encoder, cutoff=0.6):
    classes = list(encoder.classes_)
    if user_input in classes:
        return user_input
    close_matches = difflib.get_close_matches(user_input, classes, n=1, cutoff=cutoff)
    if close_matches:
        return close_matches[0]
    else:
        raise ValueError(f"Soil Type '{user_input}' not found. Available: {classes}")

# Find the correct moisture range
def find_moisture_range(user_moisture, encoder):
    matching_ranges = []
    for range_str in encoder.classes_:
        try:
            min_val, max_val = map(float, range_str.split(','))
            if min_val <= user_moisture <= max_val:
                matching_ranges.append(range_str)
        except Exception:
            continue
    if not matching_ranges:
        raise ValueError(f"No moisture range found for input moisture {user_moisture}.")
    return matching_ranges[0]

# --- Crop Recommendation Route ---
@app.route("/recommend-crop", methods=["POST"])
def recommend_crop():
    try:
        data = request.json

        # Extract inputs
        N = data["N"]
        P = data["P"]
        K = data["K"]
        temperature = data["temperature"]
        humidity = data["humidity"]
        region_input = clean_text(data["Region"])
        soil_input = clean_text(data["Soil_Type"])
        moisture_input = data["soil_moisture"]

        # Encode categorical inputs
        try:
            soil_type_match = match_soil_type(soil_input, soil_encoder_crop)
            soil_numeric = soil_encoder_crop.transform([soil_type_match])[0]
        except ValueError as e:
            return jsonify({"error": str(e)})

        try:
            region_numeric = region_encoder_crop.transform([region_input])[0]
        except ValueError:
            return jsonify({"error": f"Region '{region_input}' not found."})

        try:
            moisture_range_str = find_moisture_range(moisture_input, moisture_encoder_crop)
            moisture_numeric = moisture_encoder_crop.transform([moisture_range_str])[0]
        except ValueError as e:
            return jsonify({"error": str(e)})

        # Form feature vector
        feature_vector = pd.DataFrame([[N, P, K, temperature, humidity, region_numeric, soil_numeric, moisture_numeric]],
                                      columns=["N", "P", "K", "temperature", "humidity", "Region", "Soil_Type", "soil_moisture_range"])

        # Predict crop
        predicted_crop = crop_model.predict(feature_vector)[0]

        return jsonify({"Recommended Crop": predicted_crop})

    except Exception as e:
        return jsonify({"error": str(e)})


# --- Fertilizer Optimization Route ---
@app.route("/optimize-fertilizer", methods=["POST"])
def optimize_fertilizer():
    try:
        data = request.json

        # Extract inputs
        N = data["N"]
        P = data["P"]
        K = data["K"]
        moisture = data["soil_moisture_range"]
        temperature = data["temperature"]
        humidity = data["humidity"]
        region_input = clean_text_fertilizer(data["Region"])
        soil_input = clean_text_fertilizer(data["Soil_Type"])
        crop_input = clean_text_fertilizer(data["Crop"])

        # Encode categorical inputs
        try:
            region_encoded = region_encoder_fert.transform([region_input])[0]
        except ValueError:
            return jsonify({"error": f"Region '{region_input}' not found."})

        try:
            crop_encoded = crop_encoder_fert.transform([crop_input])[0]
        except ValueError:
            return jsonify({"error": f"Crop '{crop_input}' not found."})

        try:
            soil_type_match = match_soil_type(soil_input, soil_encoder_fert)
            soil_encoded = soil_encoder_fert.transform([soil_type_match])[0]
        except ValueError as e:
            return jsonify({"error": str(e)})

        # Create feature vector
        feature_vector = np.array([[N, P, K, moisture, temperature, humidity, region_encoded, soil_encoded, crop_encoded]])
        feature_names = ['N', 'P', 'K', 'soil_moisture_range', 'temperature', 'humidity',
                         'Region_encoded', 'Soil_Type_encoded', 'Crop_encoded']
        input_df = pd.DataFrame(feature_vector, columns=feature_names)

        # Predict fertilizer adjustments
        adj_N = model_N.predict(input_df)[0]
        adj_P = model_P.predict(input_df)[0]
        adj_K = model_K.predict(input_df)[0]

        # Interpretation function
        def interpret_adjustment(adj, nutrient):
            if adj > 0:
                return f"Increase {nutrient} by {adj:.2f} units."
            elif adj < 0:
                return f"Reduce {nutrient} by {abs(adj):.2f} units."
            else:
                return f"{nutrient} level is optimal."

        return jsonify({
            "Nitrogen": interpret_adjustment(adj_N, "Nitrogen"),
            "Phosphorus": interpret_adjustment(adj_P, "Phosphorus"),
            "Potassium": interpret_adjustment(adj_K, "Potassium")
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")  # Define the homepage route
def home():
    return "Welcome to the Crop Recommendation API!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

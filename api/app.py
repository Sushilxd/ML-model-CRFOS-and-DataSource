from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import ast
import difflib
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins
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
def match_soil_type(user_input, encoder):
    classes = list(encoder.classes_)
    user_input = clean_text(user_input).lower()  # Normalize input

    # Exact match check
    if user_input in [clean_text(cls).lower() for cls in classes]:
        return user_input

    # Partial match check
    for cls in classes:
        if user_input in clean_text(cls).lower():
            return cls

    # Fuzzy match fallback
    close_matches = difflib.get_close_matches(user_input, [clean_text(cls).lower() for cls in classes], n=1, cutoff=0.4)
    if close_matches:
        return close_matches[0]

    # If no match is found
    raise ValueError(f"Soil Type '{user_input}' not found. Available: {classes}")

# Find the correct moisture range
def find_moisture_range(user_moisture, encoder):
    matching_ranges = []

    # Ensure moisture is a float
    user_moisture = float(user_moisture)

    # Clean encoder classes
    cleaned_classes = [s.replace("'", "").strip() for s in encoder.classes_]
    print(f"Cleaned Encoder classes: {cleaned_classes}")  # Debugging output

    for range_str in cleaned_classes:
        try:
            # Ensure proper splitting & conversion
            min_str, max_str = range_str.replace(" ", "").split(',')
            min_val, max_val = float(min_str), float(max_str)  # Explicit conversion

            print(f"Checking range: {min_val} - {max_val} for moisture {user_moisture}")  # Debugging output

            if min_val <= user_moisture <= max_val:
                print(f"✅ Found range: {range_str}")  # Debugging success case
                return range_str
        except ValueError as e:
            print(f"❌ ValueError in parsing range {range_str}: {e}")  # Catch string-to-float errors
        except Exception as e:
            print(f"❌ Unexpected error parsing range {range_str}: {e}")  # Catch all errors

    print(f"❌ No range found for moisture {user_moisture}")  # Debugging output if it fails
    raise ValueError(f"No moisture range found for input moisture {user_moisture}.")


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
        Recommended_Crop = predicted_crop
        return jsonify({"Recommended_Crop": Recommended_Crop})

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

@app.route("/")  
def home():
    return "Welcome to the Crop Recommendation API!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# Soil Data API

## Overview
This API provides two main functionalities:
1. **Crop Recommendation** - Suggests the best crop based on soil and environmental conditions.
2. **Fertilizer Optimization** - Recommends adjustments for Nitrogen (N), Phosphorus (P), and Potassium (K) levels.

## Tech Stack
- **Backend:** Flask
- **Machine Learning:** scikit-learn, joblib
- **Data Processing:** NumPy, Pandas

## Project Structure
```
📂 soil-prediction-api
 ├── 📂 models
 │   ├── crop_recommendation_model.pkl
 │   ├── Region_encoder.pkl
 │   ├── Soil_Type_encoder.pkl
 │   ├── moisture_encoder.pkl
 │   ├── fert_adj_model_N.pkl
 │   ├── fert_adj_model_P.pkl
 │   ├── fert_adj_model_K.pkl
 │   ├── region_encoder_fert_adj.pkl
 │   ├── soil_encoder_fert_adj.pkl
 │   ├── crop_encoder_fert_adj.pkl
 ├── app.py
 ├── requirements.txt
 ├── Procfile
 ├── README.md
```

## Installation & Setup
### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd soil-prediction-api
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run the API Locally
```sh
python app.py
```
Server runs at `http://127.0.0.1:5000`

## API Endpoints
### 1️⃣ Crop Recommendation
**Endpoint:** `/recommend-crop`  
**Method:** `POST`  
**Description:** Predicts the best crop based on soil and environmental factors.

**Request Body (JSON):**
```json
{
    "N": 20.0,
    "P": 30.0,
    "K": 40.0,
    "temperature": 28.0,
    "humidity": 75.0,
    "Region": "Island Region (Andaman & Nicobar)",
    "Soil_Type": "Coastal Alluvium Salty",
    "soil_moisture": 80.0
}
```

**Response (JSON):**
```json
{
    "Recommended Crop": "jute"
}
```

### 2️⃣ Fertilizer Optimization
**Endpoint:** `/optimize-fertilizer`  
**Method:** `POST`  
**Description:** Recommends adjustments for N, P, and K levels.

**Request Body (JSON):**
```json
{
    "N": 20.0,
    "P": 30.0,
    "K": 40.0,
    "soil_moisture_range": 85.0,
    "temperature": 28.0,
    "humidity": 80.0,
    "Region": "Island Region (Andaman & Nicobar)",
    "Soil_Type": "Coastal Alluvium Salty",
    "Crop": "jute"
}
```

**Response (JSON):**
```json
{
    "Nitrogen": "Increase Nitrogen by 5.20 units.",
    "Phosphorus": "Reduce Phosphorus by 3.10 units.",
    "Potassium": "Potassium level is optimal."
}
```

## Deployment on Render
1. **Push to GitHub:**
```sh
git add .
git commit -m "Initial commit"
git push origin main
```
2. **Deploy on Render:**
   - Go to [Render](https://render.com/)
   - Select **New Web Service**
   - Connect your repository
   - Set **Python 3.9+** as runtime
   - Deploy 🚀

## License
MIT License


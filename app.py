from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from functools import wraps

load_dotenv()
app = Flask(__name__)
CORS(app)  

model = joblib.load('model.pkl')


atlas_uri = os.getenv('MONGODB_ATLAS_URI')
client = MongoClient(atlas_uri)
db = client["heartproblm"]
predictions_collection = db["heartdata"]

def encode_admission_type(admission_type):
    return {
        "admission_type_EMERGENCY": int(admission_type.upper() == "EMERGENCY"),
        "admission_type_URGENT": int(admission_type.upper() == "URGENT")
    }

def encode_flag(flag_value: str) -> int:
    mapping = {
        "nan": 0,
        "abnormal": 1,
        "delta": 2
    }
    return mapping.get(flag_value.lower(), 0)

def encode_discharge_location(value: str) -> dict:
    value = value.strip().upper()
    return {
        "discharge_location_HOME": int(value == "HOME"),
        "discharge_location_HOME HEALTH CARE": int(value == "HOME HEALTH CARE"),
        "discharge_location_SNF": int(value == "SNF"),
        "discharge_location_SHORT TERM HOSPITAL": int(value == "SHORT TERM HOSPITAL"),
        "discharge_location_REHAB/DISTINCT PART HOSP": int(value == "REHAB/DISTINCT PART HOSP"),
        "discharge_location_OTHER FACILITY": int(value == "OTHER FACILITY"),
    }

def get_insurance_risk(insurance_type: str) -> int:
    insurance_risk = {
        'Medicare': 3,
        'Medicaid': 4,
        'Private': 1,
        'Self Pay': 2,
        'Government': 2
    }
    return insurance_risk.get(insurance_type, 0)

def calculate_length_of_stay(admit_time_str: str, discharge_time_str: str) -> int:
    admit_time = datetime.strptime(admit_time_str, "%Y-%m-%d %H:%M:%S")
    discharge_time = datetime.strptime(discharge_time_str, "%Y-%m-%d %H:%M:%S")
    return (discharge_time - admit_time).days

def get_admit_weekday(admit_time_str: str) -> int:
    admit_time = datetime.strptime(admit_time_str, "%Y-%m-%d %H:%M:%S")
    return admit_time.weekday()
# Keep your helper functions (encode_flag, calculate_length_of_stay, etc.) unchanged

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Process input data
        ntprobnp = float(data.get('ntprobnp'))
        creatinine = float(data.get('creatinine'))
        urea_nitrogen = float(data.get('urea_nitrogen'))
        sodium = float(data.get('sodium'))
        potassium = float(data.get('potassium'))
        albumin = float(data.get('albumin'))
        crp = float(data.get('c_reactive_protein'))
        hemoglobin = float(data.get('hemoglobin'))
        hematocrit = float(data.get('hematocrit'))
        magnesium = float(data.get('magnesium'))

        flags = {
            "ntprobnp_flag": encode_flag(data.get("ntprobnp_flag")),
            "creatinine_flag": encode_flag(data.get("creatinine_flag")),
            "urea nitrogen_flag": encode_flag(data.get("urea_nitrogen_flag")),
            "sodium_flag": encode_flag(data.get("sodium_flag")),
            "potassium_flag": encode_flag(data.get("potassium_flag")),
            "albumin_flag": encode_flag(data.get("albumin_flag")),
            "c-reactive protein_flag": encode_flag(data.get("c_reactive_protein_flag")),
            "hemoglobin_flag": encode_flag(data.get("hemoglobin_flag")),
            "hematocrit_flag": encode_flag(data.get("hematocrit_flag")),
            "magnesium_flag": encode_flag(data.get("magnesium_flag")),
        }

        admission = encode_admission_type(data.get("admission_type"))
        discharge = encode_discharge_location(data.get("discharge_location"))
        insurance_risk = get_insurance_risk(data.get("insurance"))

        length_of_stay = calculate_length_of_stay(
            data.get("admit_time"), 
            data.get("discharge_time")
        )
        admit_weekday = get_admit_weekday(data.get("admit_time"))

        # Prepare features for prediction
        feature_order = [
            'ntprobnp', 'ntprobnp_flag', 'creatinine', 'creatinine_flag',
            'urea nitrogen', 'urea nitrogen_flag', 'sodium', 'sodium_flag',
            'potassium', 'potassium_flag', 'albumin', 'albumin_flag',
            'c-reactive protein', 'c-reactive protein_flag', 'hemoglobin', 'hemoglobin_flag',
            'hematocrit', 'hematocrit_flag', 'magnesium', 'magnesium_flag',
            'admission_type_EMERGENCY', 'admission_type_URGENT',
            'discharge_location_HOME', 'discharge_location_HOME HEALTH CARE',
            'discharge_location_SNF', 'discharge_location_SHORT TERM HOSPITAL',
            'discharge_location_REHAB/DISTINCT PART HOSP', 'discharge_location_OTHER FACILITY',
            'insurance_risk', 'length_of_stay', 'admit_weekday'
        ]

        result = {
            "patient_id": data.get("patient_id"),
            "patient_name": data.get("patient_name"),
            "ntprobnp": ntprobnp,
            "creatinine": creatinine,
            "urea_nitrogen": urea_nitrogen,
            "sodium": sodium,
            "potassium": potassium,
            "albumin": albumin,
            "c_reactive_protein": crp,
            "hemoglobin": hemoglobin,
            "hematocrit": hematocrit,
            "magnesium": magnesium,
            **flags,
            **admission,
            **discharge,
            "insurance": data.get("insurance"),
            "insurance_risk": insurance_risk,
            "length_of_stay": length_of_stay,
            "admit_weekday": admit_weekday,
            "admission_type": data.get("admission_type"),
            "discharge_location": data.get("discharge_location"),
            "admit_time": data.get("admit_time"),
            "discharge_time": data.get("discharge_time"),
            "timestamp": datetime.now()
        }

        # Make prediction
        X = np.array([result.get(f, 0) for f in feature_order]).reshape(1, -1)
        prediction = model.predict(X)[0]
        probability = float(model.predict_proba(X)[0][1])

        # Add prediction results
        result.update({
            "prediction_result": "Yes" if prediction == 1 else "No",
            "readmission_probability": probability
        })

        # Store in MongoDB
        predictions_collection.insert_one(result)

        return jsonify({
            'prediction': int(prediction),
            'probability': probability,
            'message': 'Prediction successful'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
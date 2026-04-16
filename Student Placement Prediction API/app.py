from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("target_encoder.pkl", "rb") as f:
        target_encoder = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    print("✅ Model and encoders loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    label_encoders = None
    target_encoder = None
    feature_names = None

@app.route('/')
def home():
    return jsonify({
        "message": "Student Placement Prediction API",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Make prediction with student data",
            "/features": "GET - Get required features"
        }
    })

@app.route('/features')
def get_features():
    if feature_names is None:
        return jsonify({"error": "Model not loaded"}), 500

    return jsonify({
        "features": feature_names,
        "categorical_features": list(label_encoders.keys()),
        "example": {
            "gender": "M/F",
            "ssc_p": 85.5,
            "ssc_b": "Central/Others",
            "hsc_p": 78.2,
            "hsc_b": "Central/Others",
            "hsc_s": "Commerce/Science/Arts",
            "degree_p": 75.8,
            "degree_t": "Sci&Tech/Comm&Mgmt/Others",
            "workex": "Yes/No",
            "etest_p": 82.3,
            "specialisation": "Mkt&HR/Mkt&Fin",
            "mba_p": 68.9
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoders is None or target_encoder is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Prepare features in correct order
        features = []
        for feature in feature_names:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

            value = data[feature]

            # Encode categorical features
            if feature in label_encoders:
                if value not in label_encoders[feature].classes_:
                    return jsonify({"error": f"Invalid value '{value}' for {feature}. Valid values: {list(label_encoders[feature].classes_)}"}), 400
                encoded_value = label_encoders[feature].transform([value])[0]
                features.append(encoded_value)
            else:
                # Numerical features
                try:
                    features.append(float(value))
                except ValueError:
                    return jsonify({"error": f"Invalid numerical value for {feature}: {value}"}), 400

        # Make prediction
        features_array = np.array([features])
        prediction_encoded = model.predict(features_array)[0]
        prediction_proba = model.predict_proba(features_array)[0]

        # Decode prediction
        prediction = target_encoder.inverse_transform([prediction_encoded])[0]

        # Get confidence scores
        confidence_scores = {
            target_encoder.classes_[i]: float(prediction_proba[i])
            for i in range(len(target_encoder.classes_))
        }

        return jsonify({
            "prediction": prediction,
            "confidence": confidence_scores,
            "input_features": data
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
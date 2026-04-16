import streamlit as st
import pickle
import numpy as np
import requests

st.title("🎓 Student Placement Prediction System")

# Load model and encoders
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        with open("target_encoder.pkl", "rb") as f:
            target_encoder = pickle.load(f)
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        return model, label_encoders, target_encoder, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

model, label_encoders, target_encoder, feature_names = load_model()

if model is None:
    st.error("Model not loaded. Please run train_model.py first.")
    st.stop()

st.markdown("---")

# Input form
st.subheader("Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["M", "F"])
    ssc_p = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, step=0.1)
    ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
    hsc_p = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, step=0.1)
    hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
    hsc_s = st.selectbox("HSC Stream", ["Commerce", "Science", "Arts"])
    degree_p = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, step=0.1)

with col2:
    degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
    workex = st.selectbox("Work Experience", ["Yes", "No"])
    etest_p = st.number_input("Entrance Test Percentage", min_value=0.0, max_value=100.0, step=0.1)
    specialisation = st.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])
    mba_p = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0, step=0.1)

# Predict button
if st.button("🔮 Predict Placement", type="primary"):
    # Prepare input data
    input_data = {
        "gender": gender,
        "ssc_p": ssc_p,
        "ssc_b": ssc_b,
        "hsc_p": hsc_p,
        "hsc_b": hsc_b,
        "hsc_s": hsc_s,
        "degree_p": degree_p,
        "degree_t": degree_t,
        "workex": workex,
        "etest_p": etest_p,
        "specialisation": specialisation,
        "mba_p": mba_p
    }

    # Make prediction locally
    try:
        # Prepare features in correct order
        features = []
        for feature in feature_names:
            value = input_data[feature]
            if feature in label_encoders:
                encoded_value = label_encoders[feature].transform([value])[0]
                features.append(encoded_value)
            else:
                features.append(float(value))

        # Make prediction
        features_array = np.array([features])
        prediction_encoded = model.predict(features_array)[0]
        prediction_proba = model.predict_proba(features_array)[0]

        # Decode prediction
        prediction = target_encoder.inverse_transform([prediction_encoded])[0]

        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        if prediction == "Placed":
            st.success(f"🎉 **{prediction}**")
        else:
            st.error(f"❌ **{prediction}**")

        # Confidence scores
        st.subheader("Confidence Scores")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Placed", f"{prediction_proba[1]*100:.1f}%")

        with col2:
            st.metric("Not Placed", f"{prediction_proba[0]*100:.1f}%")

        # Show input summary
        st.subheader("📋 Input Summary")
        st.json(input_data)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit & Scikit-learn*")
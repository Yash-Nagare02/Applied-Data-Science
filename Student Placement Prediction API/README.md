# Student Placement Prediction API

A machine learning project that predicts student placement based on academic performance and other factors using Flask API and Streamlit web interface.

## 📊 Dataset

The project uses the "Placement_Data_Full_Class.csv" dataset containing student information including:
- Academic percentages (SSC, HSC, Degree, MBA)
- Board types and streams
- Work experience
- Entrance test scores
- MBA specialisation
- Placement status (target variable)

## 🚀 Features

- **Flask API**: RESTful API for placement prediction
- **Streamlit App**: User-friendly web interface
- **Machine Learning**: Random Forest classifier
- **Data Preprocessing**: Automatic encoding of categorical variables
- **Input Validation**: Comprehensive error handling

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model:**
   ```bash
   python train_model.py
   ```

## 📱 Usage

### Flask API

**Start the API server:**
```bash
python app.py
```

**API Endpoints:**

- `GET /` - API status and available endpoints
- `GET /features` - Get required features and example input
- `POST /predict` - Make placement prediction

**Example API Request:**
```python
import requests

data = {
    "gender": "M",
    "ssc_p": 85.5,
    "ssc_b": "Central",
    "hsc_p": 78.2,
    "hsc_b": "Central",
    "hsc_s": "Commerce",
    "degree_p": 75.8,
    "degree_t": "Comm&Mgmt",
    "workex": "Yes",
    "etest_p": 82.3,
    "specialisation": "Mkt&Fin",
    "mba_p": 68.9
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
```

**Test the API:**
```bash
python test_api.py
```

### Streamlit Web App

**Start the web interface:**
```bash
streamlit run streamlit_app.py
```

The web app will open in your browser with an intuitive form for entering student details.

## 📁 Project Structure

```
├── app.py                 # Flask API server
├── streamlit_app.py       # Streamlit web interface
├── train_model.py         # Model training script
├── test_api.py           # API testing script
├── requirements.txt       # Python dependencies
├── Placement_Data_Full_Class.csv  # Dataset
├── model.pkl             # Trained model
├── label_encoders.pkl    # Categorical encoders
├── target_encoder.pkl    # Target encoder
└── feature_names.pkl     # Feature names
```

## 🔧 API Response Format

**Success Response:**
```json
{
  "prediction": "Placed",
  "confidence": {
    "Not Placed": 0.23,
    "Placed": 0.77
  },
  "input_features": {
    "gender": "M",
    "ssc_p": 85.5,
    ...
  }
}
```

**Error Response:**
```json
{
  "error": "Missing feature: gender"
}
```

## 📈 Model Performance

The Random Forest classifier is trained on the placement dataset and provides:
- Binary classification (Placed/Not Placed)
- Confidence scores for predictions
- Feature importance analysis

## 🤝 Contributing

Feel free to improve the model, add more features, or enhance the API!

## 📄 License

This project is for educational purposes.
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load dataset
df = pd.read_csv("Placement_Data_Full_Class.csv")

# Drop unnecessary columns
df = df.drop(['sl_no', 'salary'], axis=1)  # sl_no is ID, salary has NaN for not placed

# Handle missing values (though there shouldn't be any in this dataset)
df.dropna(inplace=True)

# Identify categorical columns
categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']

# Create and fit label encoders
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
X = df.drop('status', axis=1)
y = df['status']

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y_encoded)

# Save model and encoders
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

# Save feature names for reference
feature_names = list(X.columns)
with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("Model and encoders saved successfully!")
print(f"Features: {feature_names}")
print(f"Target classes: {target_encoder.classes_}")
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_insurance_claim_model.pkl")

# Example new data (must match training features!)
new_data = pd.DataFrame([{
    "Age": 35,
    "Duration": 10,
    "Destination_Europe": 1,
    "Destination_Asia": 0,
    "EmploymentType_Salaried": 1,
    # ... include all features from training
}])

# Predict
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)[:,1]

print("Predicted Claim:", prediction[0])
print("Probability of Claim:", probability[0])

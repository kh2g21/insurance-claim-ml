import pandas as pd
import joblib

# Load trained model and training columns
model, train_cols = joblib.load("best_insurance_claim_model.pkl")

print(type(model))


# Collect user input
agency = input("Enter Agency (e.g., CBH, CWT): ")
agency_type = input("Enter Agency Type (e.g., Travel Agency, Online): ")
channel = input("Enter Distribution Channel (Offline/Online): ")
product = input("Enter Product Name (e.g., Comprehensive Plan): ")
duration = int(input("Enter Duration (days): "))
destination = input("Enter Destination (e.g., MALAYSIA, AUSTRALIA): ")
net_sales = float(input("Enter Net Sales: "))
commission = float(input("Enter Commission (in value): "))
gender = input("Enter Gender (M/F): ")
age = int(input("Enter Age: "))

# Build dataframe
new_data = pd.DataFrame([{
    "Agency": agency,
    "Agency Type": agency_type,
    "Distribution Channel": channel,
    "Product Name": product,
    "Duration": duration,
    "Destination": destination,
    "Net Sales": net_sales,
    "Commision (in value)": commission,
    "Gender": gender,
    "Age": age
}])

# Apply preprocessing
new_data = pd.get_dummies(new_data, drop_first=True)

# Reindex to training columns (handle missing values)
new_data = new_data.reindex(columns=train_cols, fill_value=0)

# Predict
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)[:, 1]

print("\nPredicted Claim:", "Yes" if prediction[0] == 1 else "No")
print("Probability of Claim:", round(probability[0], 4))

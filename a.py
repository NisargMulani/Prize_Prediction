import joblib
import pandas as pd

# Load model
model = joblib.load("model.joblib")

# Prepare a sample input
sample_input = pd.DataFrame([{
    'main_category': 0,
    'sub_category': 3,
    'actual_price': 1799,
    'ratings': 4.5,
    'no_of_ratings': 250
}])

# Predict
predicted_price = model.predict(sample_input)[0]
print("Predicted Discount Price:", round(predicted_price, 2))
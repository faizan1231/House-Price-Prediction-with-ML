import pandas as pd
import joblib
from django.shortcuts import render

# Load the trained model
model = joblib.load("C:/Users/Ali Computer/House Pridiction/house_price_model.pkl")

# Load dataset to get feature names
df = pd.read_csv("C:/Users/Ali Computer/House Pridiction/House-Price-Prediction-clean.csv")
feature_columns = df.drop(columns=["Id", "SalePrice"]).columns  # Feature names

def predict_price(request):
    predicted_price = None

    if request.method == "POST":
        try:
            # Get input values dynamically
            input_data = [float(request.POST[col]) for col in feature_columns]
            input_df = pd.DataFrame([input_data], columns=feature_columns)
            
            # Make prediction
            predicted_price = model.predict(input_df)[0]

        except Exception as e:
            predicted_price = f"Error: {e}"

    return render(request, "home.html", {"feature_columns": feature_columns, "predicted_price": predicted_price})

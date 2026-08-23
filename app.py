import streamlit as st
import joblib
import pandas as pd

model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")
st.write("Enter the house details to predict the estimated price.")

area = st.number_input("House Area", min_value=0.0, value=2000.0)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, value=2)
age = st.number_input("House Age", min_value=0, value=5)

if st.button("Predict Price"):
    new_house = pd.DataFrame(
        [[area, bedrooms, bathrooms, age]],
        columns=["area", "bedrooms", "bathrooms", "age"]
    )

    predicted_price = model.predict(new_house)

    st.success(f"Estimated House Price: ₹ {predicted_price[0]:,.2f}")
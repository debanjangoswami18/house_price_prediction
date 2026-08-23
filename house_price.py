import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house_data.csv")

X = df[["area", "bedrooms", "bathrooms", "age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Model trained successfully!")
print("Model Accuracy:", round(accuracy * 100, 2), "%")


area = int(input("Enter house area: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
age = int(input("Enter house age: "))

new_house = pd.DataFrame(
    [[area, bedrooms, bathrooms, age]],
    columns=["area", "bedrooms", "bathrooms", "age"]
)

predicted_price = model.predict(new_house)

print("Predicted House Price: ₹", round(predicted_price[0], 2))
import joblib

joblib.dump(model, "house_price_model.pkl")
print("Model saved successfully!")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv("car_prices.csv")

print(df.head())
X = df[["Year", "Mileage", "EngineSize", "Horsepower"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel Performance")
print("-----------------------")
print("MAE :", mean_absolute_error(y_test, predictions))
print("MSE :", mean_squared_error(y_test, predictions))
print("R²  :", r2_score(y_test, predictions))

coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nCoefficients")
print(coef)
print("\nIntercept")
print(model.intercept_)

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions.round(2)
})

print("\nActual vs Predicted")
print(comparison)

new_car = pd.DataFrame({
    "Year": [2022],
    "Mileage": [15000],
    "EngineSize": [2.0],
    "Horsepower": [180]
})

price = model.predict(new_car)

print("\nPredicted Price")
print(f"${price[0]:,.2f}")

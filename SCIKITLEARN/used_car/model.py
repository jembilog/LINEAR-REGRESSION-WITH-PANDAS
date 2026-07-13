import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("used_car.csv")

df["Brand"] = df["Brand"].str.strip().str.lower()
df["FuelType"] = df["FuelType"].str.strip().str.lower()
df["Transmission"] = df["Transmission"].str.strip().str.lower()
df["EngineSize"] = df["EngineSize"].abs()

df["Price"] = df["Price"].str.replace(",", "", regex=False).str.replace("₱","",regex=False)
df["Price"] = df["Price"].astype(int)

df["Brand"] = df["Brand"].fillna(df["Brand"].mode()[0])
df["FuelType"] = df["FuelType"].fillna(df["Brand"].mode()[0])
df["Transmission"]  = df["Transmission"].fillna(df["Transmission"].mode()[0])
df["Mileage"] = df["Mileage"].fillna(df["Mileage"].median())
df["EngineSize"] = df["EngineSize"].fillna(df["EngineSize"].median())

df = df.drop_duplicates()

df = pd.get_dummies(
    df,
    columns=["Brand","FuelType","Transmission"],
    drop_first=True,
    dtype=int
)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("MAE :", mean_absolute_error(y_test, predictions))
print("MSE :", mean_squared_error(y_test, predictions))
print("R²  :", r2_score(y_test, predictions))

coefficients = pd.DataFrame({
    "Feature" : X.columns,
    "Coefficient" : model.coef_
})
print(coefficients)
print("\nIntercept")
print(model.intercept_)

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions.round(2)
})
print(comparison)

# Predict new used car price
# new_used_car = pd.DataFrame({
#     "Year": [2022],
#     "Mileage": [20000],
#     "EngineSize": [2.0],
#     "Horsepower": [170],
#     "Owners": [1],

#     "Brand_ford": [0],
#     "Brand_honda": [0],
#     "Brand_hyundai": [0],
#     "Brand_isuzu": [0],
#     "Brand_kia": [0],
#     "Brand_mazda": [0],
#     "Brand_mitsubishi": [0],
#     "Brand_nissan": [0],
#     "Brand_suzuki": [0],
#     "Brand_toyota": [1],

#     "FuelType_petrol": [1],

#     "Transmission_manual": [0]
# })

new_used_car = pd.DataFrame(0, index=[0], columns=X.columns)
new_used_car["Year"] = 2022
new_used_car["Mileage"] = 20000
new_used_car["EngineSize"] = 2.0
new_used_car["Horsepower"] = 170
new_used_car["Owners"] = 1

new_used_car["Brand_toyota"] = 1
new_used_car["FuelType_petrol"] = 1
new_used_car["Transmission_manual"] = 0 

predicted_price = model.predict(new_used_car)

print("\n" + "=" * 50)
print("NEW USED CAR PREDICTION")
print("=" * 50)

print(new_used_car)

print(f"\nPredicted Price: ₱{predicted_price[0]:,.2f}")

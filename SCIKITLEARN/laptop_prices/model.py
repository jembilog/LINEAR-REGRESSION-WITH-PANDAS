import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD DATASET

df = pd.read_csv("laptop_prices.csv")

df["Brand"] = df["Brand"].str.strip()
df["CPU"] = df["CPU"].str.strip()
df["GPU"] = df["GPU"].str.strip()

df["Brand"] = df["Brand"].str.lower()
df["CPU"] = df["CPU"].str.lower()
df["GPU"] = df["GPU"].str.lower()

df["Price"] = df["Price"].str.replace(",", "", regex=False)
df["Price"] = df["Price"].astype(int)

df["RAM"] = df["RAM"].fillna(df["RAM"].median())
df["SSD"] = df["SSD"].fillna(df["SSD"].median())

df["Brand"] = df["Brand"].fillna(df["Brand"].mode()[0])
df["CPU"] = df["CPU"].fillna(df["CPU"].mode()[0])
df["GPU"] = df["GPU"].fillna(df["GPU"].mode()[0])

# REMOVE DUPLICATES

df = df.drop_duplicates()

print("AFTER CLEANING")


print(df.head())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

df = pd.get_dummies(
    df,
    columns=["Brand", "CPU", "GPU"],
    drop_first=True,
    dtype=int
)

print("\n" + "=" * 50)
print("ENCODED DATASET")
print("=" * 50)
print(df.head())

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print("MAE :", mean_absolute_error(y_test, predictions))
print("MSE :", mean_squared_error(y_test, predictions))
print("R²  :", r2_score(y_test, predictions))

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("FEATURE COEFFICIENTS")
print(coefficients)

print("\nIntercept")
print(model.intercept_)

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions.round(2)
})

print("ACTUAL VS PREDICTED")
print(comparison)


# Note: acer and i5 are dropped (first alphabetically) due to drop_first=True
# integrated GPU is also dropped
new_laptop = pd.DataFrame({
    "RAM": [16],
    "SSD": [512],
    "ScreenSize": [15.6],
    "Weight": [2.2],

    "Brand_asus": [1],
    "Brand_dell": [0],
    "Brand_hp": [0],
    "Brand_lenovo": [0],
    "Brand_msi": [0],

    "CPU_i7": [0],
    "CPU_i9": [0],
    "CPU_ryzen5": [0],
    "CPU_ryzen7": [1],

    "GPU_rtx3050": [0],
    "GPU_rtx3060": [0],
    "GPU_rtx4060": [1],
    "GPU_rtx4070": [0],
    "GPU_rtx4080": [0]
})

predicted_price = model.predict(new_laptop)

print("\n" + "=" * 50)
print("NEW LAPTOP PREDICTION")
print("=" * 50)

print(new_laptop)

print(f"\nPredicted Price: PHP {predicted_price[0]:,.2f}")

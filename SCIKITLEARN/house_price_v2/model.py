import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("house_price_v2.csv")
df["Area"] = df["Area"].fillna(df["Area"].median())
# df["Bedrooms"] = df["Bedrooms"].fillna(df["Bedrooms"].median())
# df["Bathrooms"] = df["Bathrooms"].fillna(df["Bathrooms"].median())
# df["Garage"] = df["Garage"].fillna(df["Garage"].mode()[0])
# df["Location"] = df["Location"].fillna(df["Location"].mode()[0])

fill_columns_median = ["Area", "Bedrooms", "Bathrooms"]
for col in fill_columns_median:
    df[col] = df[col].fillna(df[col].median())
fill_columns_mode = ["Garage", "Location"]
for col in fill_columns_mode:
    df[col] = df[col].fillna(df[col].mode()[0])
absolute_columns = ["Area", "Bedrooms", "Floors", "HouseAge"]
for col in absolute_columns:
    df[col] = df[col].abs()
cap_columns = ["Garage", "Furnished", "Location"]
for col in cap_columns:
    df[col] = df[col].str.strip().str.lower().str.replace("city","",regex=True).str.strip()
df["Price"] = df["Price"].str.replace(",","",regex=False).str.replace("₱","",regex=False).astype(int)

df = df.drop_duplicates()

df = pd.get_dummies(df, columns=["Location" ,"Garage", "Furnished"],drop_first=True,dtype=int)
# le  =LabelEncoder()
# le_target = ["Garage", "Furnished"]


X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
print("\nCoefficients")
print(coefficients)
print("\nIntercept")
print(model.intercept_)

comparison = pd.DataFrame({
    "Actual" : y_test.values,
    "Predicted" : predictions.round(2)
})
print(comparison)

new_house = pd.DataFrame(0,index=[0], columns=X.columns)
new_house["Area"] = 170
new_house["Bedrooms"] = 5
new_house["Bathrooms"] = 3
new_house["Floors"] = 4
new_house["Garage_yes"] = 1
new_house["Furnished_yes"] = 2

new_house["Location_manila"] = 1
new_house["HouseAge"] = 2

predicted_price = model.predict(new_house)
print(new_house)
print(f'\nHouse Price: ₱{predicted_price[0]:,.2f}')

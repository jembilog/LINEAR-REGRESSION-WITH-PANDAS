import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("medical_insurance.csv")

df["Smoker"] = df["Smoker"].map({
    "No" : 0,
    "Yes" : 1
})

X = df[["Age", "BMI", "Children", "Smoker"]]
y = df["InsuranceCost"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nModel Performance")
print("MAE :", mean_absolute_error(y_test, predictions))
print("MSE :", mean_squared_error(y_test, predictions))
print("R²  :", r2_score(y_test, predictions))

coef = pd.DataFrame({
    "Feature" : X.columns,
    "Coefficient" : model.coef_
})

print("\nCoefficientss")
print(coef)
print("\nBias")
print(model.intercept_)

comparison = pd.DataFrame({
    "Actual" : y_test.values,
    "Predicted" : predictions.round(2)
})

print("\nActual vs Predicted")
print(comparison)

new_person = pd.DataFrame({
    "Age" : [32],
    "BMI" : [28.5],
    "Children" : [1],
    "Smoker" : [0]
})

predicted_cost  = model.predict(new_person)
print("\nPredicted Insurance Cost")
print(f"${predicted_cost[0]:,.2f}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

df = pd.read_csv("employee_salary.csv")
#print(df)
df = pd.get_dummies(
    df,
    columns=["Education", "JobRole"],
    drop_first=True
)

X = df.drop("Salary", axis=1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\nModel Performance")
print("---------------------")
print("MAE :", mean_absolute_error(y_test, predictions))
print("MSE :", mean_squared_error(y_test, predictions))
print("R²  :", r2_score(y_test, predictions))

coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients")
print(coef)

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions.round(2)
})

print("\nActual vs Predicted")
print(comparison)

new_employee = pd.DataFrame({
    "Age": [30],
    "Experience": [6],
    "Education_High School": [False],
    "Education_Master": [True],
    "JobRole_Developer": [True],
    "JobRole_Manager": [False]
})

predicted_salary = model.predict(new_employee)

print("\nPredicted Salary")
print("----------------")
print(f"${predicted_salary[0]:,.2f}")

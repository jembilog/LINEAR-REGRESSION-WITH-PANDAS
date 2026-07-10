import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("student_score.csv")

X = df[["HoursStudied"]]
y = df["ExamScore"]

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)

model  = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test,predictions)
mse = mean_squared_error(y_test,predictions)
r2 = r2_score(y_test,predictions)

print("\nModel Evaluation")
print("MAE :", mae)
print("MSE :", mse)
print("R²  :", r2)
print("\nModel Parameters")
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

comparison = pd.DataFrame({
    "Actual" : y_test.values,
    "Predicted" : predictions
})
print("\nActual vs Predicted")
print(comparison)

new_student = pd.DataFrame({
    "HoursStudied" : [6.5]
})
predicted_score = model.predict(new_student)
print("\nPrediction")
print("Hours Studied:", new_student.iloc[0]["HoursStudied"])
print("Predicted Exam Score:", predicted_score[0])

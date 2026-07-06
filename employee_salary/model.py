import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("employee_salary_dataset.csv")
print(df.isnull().sum())
#preprocessing
df["EducationYears"] = df["EducationYears"].fillna(df["EducationYears"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
print(df.isnull().sum())

X = df[["Experience", "EducationYears", "Age", "ProjectsCompleted"]].values
Y = df["Salary"].values
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Y_mean = np.mean(Y)
# Y_std = np.std(Y)
# Y = (Y - Y_mean) / Y_std
w = np.zeros(4)
b = 0
learning_rate = 0.001
epochs = 5000
n = len(X)
losses = []

for epoch in range(epochs):
    Y_pred = np.dot(X, w) + b
    mse = np.mean((Y - Y_pred) ** 2)
    losses.append(mse)

    dw = (2/n) * np.dot(X.T, (Y_pred - Y))
    db = (2/n) * np.sum(Y_pred - Y)
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {mse:.4f}")
# Final prediction
Y_pred = np.dot(X, w) + b

# Metrics
mse = np.mean((Y - Y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y - Y_pred))

ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_residual = np.sum((Y - Y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

new_input = np.array([
    8,
    15,
    38,
    6
])
new_input_scaled = (new_input - X_mean) / X_std
new_output = np.dot(new_input_scaled, w) + b
print(new_output)
print(r2)
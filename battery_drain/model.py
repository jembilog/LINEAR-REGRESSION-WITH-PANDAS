import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("smartphone_battery_dataset.csv")

# print("====== ORIGINAL DATA ======")
# print(df.head())

# print("\n====== MISSING VALUES ======")
# print(df.isnull().sum())

X = df[["ScreenTime" ,"Brightness", "CPUUsage", "Temperature"]].values
Y = df["BatteryDrain"].values

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

Y_mean = np.mean(Y)
Y_std = np.std(Y)
Y = (Y - Y_mean) / Y_std

n = len(X)
w = np.zeros(4)
b = 0
learning_rate = 0.001
epochs  = 5000
losses = []

for epoch in range(epochs):
    Y_pred = np.dot(X, w) + b
    mse = np.mean((Y - Y_pred) ** 2)
    losses.append(mse)

    dw = (-2/n) * np.dot(X.T, (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db

    print(
        f"MSE: {mse:.4f}"
    )
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load csv heheh
df = pd.read_csv("solar_data.csv")

X = df["Light_Intensity"].values
Y = df["Generated_Power"].values

w = 0
b = 0
learning_rate = 0.000001
epochs = 10000
n = len(X)
loss_history = []

for epoch in range(epochs):
    Y_pred = w * X + b
    mse = np.mean((Y - Y_pred) ** 2)
    loss_history.append(mse)

    dw = (-2/n) * np.sum(X * (Y - Y_pred))
    db = (-2/n) * np.sum(Y  - Y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db

Y_pred = w * X + b
next_feature = 1320
new_prediction = w * next_feature + b

#metrics
mse = np.mean((Y - Y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y - Y_pred))

ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_residual = np.sum((Y - Y_pred) ** 2)

r2 = 1 - (ss_residual / ss_total)

#results 
print("===== MODEL =====")
print(f"Weight: {w:.6f}")
print(f"Bias: {b:.6f}")

#metrics
print("\n===== METRICS =====")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")

print(f"Predicted Generatede Power: {new_prediction:.2f}")


# plt.figure(figsize=(8,5))
# plt.scatter(X, Y, label ='Actual Data')
# plt.xlabel("Light Intensity")
# plt.ylabel("Generated Power")
# plt.title("Linear Regression")
# plt.legend()
# plt.grid(True)
# plt.show()

plt.scatter(X, Y, label='Actual Data')
# start_point= min(min(X), X_new)
# X_range = np.linspace(start_point,max(X) + 1, 100)
X_range = np.linspace(min(X), max(max(X), next_feature), 100)
Y_range = w * X_range + b
plt.plot(X_range,Y_range, color='green',label='Best fit line')
plt.scatter([next_feature],[w * next_feature + b], color='red',marker='*',label='Target')
plt.title("Solar Model")
plt.xlabel("Light Intensity")
plt.ylabel("Generated Power")
plt.grid(True)
plt.legend()
plt.show()




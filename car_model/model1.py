import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("car_mpg_dataset.csv")

X  = df[["Weight","Displacement", "Horsepower", "Cylinders"]].values
Y = df["MPG"].values

w = np.zeros(4)
b = 0
n = len(X)
learning_rate = 0.000001
epochs = 5000

losses = []

for epoch in range(epochs):
    Y_pred = np.dot(X, w) + b

    mse = np.mean((Y - Y_pred) ** 2)
    losses.append(mse)

    dw = (-2/n) * np.dot(X.T, (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db
#final prediction 
Y_pred = np.dot(X, w) + b
#new prediction
new_car = np.array([3.4, 210, 140, 6])
prediction = np.dot(new_car, w) + b

# Evaluation Metrics
mse = np.mean((Y - Y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y - Y_pred))
ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_residual = np.sum((Y - Y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

#model
print("========== MODEL ==========")
print(f"Weight Coefficient       : {w[0]:.4f}")
print(f"Displacement Coefficient : {w[1]:.4f}")
print(f"Horsepower Coefficient   : {w[2]:.4f}")
print(f"Cylinder Coefficient     : {w[3]:.4f}")
print(f"Bias                     : {b:.4f}")
print()
print("Predicted Fuel Economy")
print(f"Weight       : {new_car[0]} (1000 lbs)")
print(f"Displacement : {new_car[1]} cu in")
print(f"Horsepower   : {new_car[2]} HP")
print(f"Cylinders    : {int(new_car[3])}")
print(f"Predicted MPG: {prediction:.2f}")
print("\n========== METRICS ==========")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
print("\n========== ACTUAL vs PREDICTED ==========")
for actual, predicted in zip(Y, Y_pred):
    print(f"Actual: {actual:.2f} MPG\tPredicted: {predicted:.2f} MPG")

plt.figure(figsize=(8,5))
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))
plt.plot(Y, marker='o', label="Actual")
plt.plot(Y_pred, marker='x', label="Predicted")
plt.title("Actual vs Predicted Fuel Economy")
plt.xlabel("Car Sample")
plt.ylabel("Miles Per Gallon (MPG)")
plt.legend()
plt.grid(True)
plt.show()
#or
#plt.figure(figsize=(6,6))
# plt.scatter(Y, Y_pred)

# plt.xlabel("Actual MPG")
# plt.ylabel("Predicted MPG")
# plt.title("Actual vs Predicted")

# plt.grid(True)
# plt.show()

#if u want to look for each scatter plots because we can't plot them once in 2d, we will scatter plot per feature
plt.figure(figsize=(6,4))
plt.scatter(X[:,0], Y)
plt.xlabel("Weight")
plt.ylabel("MPG")
plt.title("Weight vs MPG")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(X[:,1], Y)
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Displacement vs MPG")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(X[:,2], Y)
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Horsepower vs MPG")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(X[:,3], Y)
plt.xlabel("Cylinders")
plt.ylabel("MPG")
plt.title("Cylinders vs MPG")
plt.grid(True)
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#DATASET
df = pd.read_csv("car_mpg_dataset.csv")

X = df[["Weight", "Displacement", "Horsepower", "Cylinders"]].values
Y = df["MPG"].values


#FEATURE SCALING
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X = (X - X_mean) / X_std


#INITIALIZE MODEL
w = np.zeros(X.shape[1])
b = 0

n = len(X)

learning_rate = 0.01
epochs = 5000

losses = []


#GRADIENT DESCENT
for epoch in range(epochs):

    Y_pred = np.dot(X, w) + b
    mse = np.mean((Y - Y_pred) ** 2)
    losses.append(mse)
    dw = (-2/n) * np.dot(X.T, (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db

#FINAL MODEL 
Y_pred = np.dot(X, w) + b

#NEW CAR PREDICTION
new_car = np.array([
    3.4,     # Weight (1000 lbs)
    210,     # Displacement
    140,     # Horsepower
    6        # Cylinders
])


# Scale new input
new_car_scaled = (new_car - X_mean) / X_std
prediction = np.dot(new_car_scaled, w) + b



#EVALUATION 
mse = np.mean((Y - Y_pred)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y - Y_pred))
ss_total = np.sum((Y - np.mean(Y))**2)
ss_residual = np.sum((Y - Y_pred)**2)
r2 = 1 - (ss_residual / ss_total)



#OUTPUT
print("========== MODEL ==========")
print(f"Weight Coefficient       : {w[0]:.4f}")
print(f"Displacement Coefficient : {w[1]:.4f}")
print(f"Horsepower Coefficient   : {w[2]:.4f}")
print(f"Cylinder Coefficient     : {w[3]:.4f}")
print(f"Bias                     : {b:.4f}")
print("\n========== NEW CAR ==========")
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
print("\n========== ACTUAL VS PREDICTED ==========")

for actual, predicted in zip(Y, Y_pred):
    print(
        f"Actual: {actual:.2f} MPG\tPredicted: {predicted:.2f} MPG"
    )



#LOSS GRAPH
plt.figure(figsize=(8,5))
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()



#ACTUAL VS PREDICTED LINE GRAPH
plt.figure(figsize=(8,5))
plt.plot(Y, marker="o", label="Actual")
plt.plot(Y_pred, marker="x", label="Predicted")
plt.title("Actual vs Predicted MPG")
plt.xlabel("Car Sample")
plt.ylabel("MPG")
plt.legend()
plt.grid(True)
plt.show()



#ACTUAL VS PREDICTED SCATTER
# plt.figure(figsize=(6,6))
# plt.scatter(Y, Y_pred)
# plt.xlabel("Actual MPG")
# plt.ylabel("Predicted MPG")
# plt.title("Actual vs Predicted MPG")
# plt.grid(True)
# plt.show()

#FEATURE SCATTER PLOTS
features = [
    "Weight",
    "Displacement",
    "Horsepower",
    "Cylinders"
]

for i in range(len(features)):

    plt.figure(figsize=(6,4))

    plt.scatter(
        df[features[i]],
        Y
    )
    plt.xlabel(features[i])
    plt.ylabel("MPG")
    plt.title(f"{features[i]} vs MPG")
    plt.grid(True)
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#DATASET
df = pd.read_csv("solar_power_dataset.csv")

X = df[["LightIntensity", "PanelTemperature","Voltage", "Current"]].values
Y = df["GeneratedPower"].values

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

#gradient descent
for epoch in range(epochs):
    Y_pred = np.dot(X, w) + b
    mse = np.mean((Y - Y_pred) ** 2)
    losses.append(mse)
    dw = (-2/n) * np.dot(X.T, (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db

#final model
Y_pred = np.dot(X, w) + b
new_input = np.array([
    800,     # Light Intensity (W/m^2)
    35,      # Panel Temperature (°C)
    30,      # Voltage (V)
    5        # Current (A)
])

# Scale new input
new_input_scaled = (new_input - X_mean) / X_std
new_output = np.dot(new_input_scaled, w) + b
# print(f"Predicted Generated Power: {new_output:.2f} Watts")

#evaluation
mse = np.mean((Y - Y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y - Y_pred))
ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_res = np.sum((Y - Y_pred) ** 2)
r_squared = 1 - (ss_res / ss_total)

#output
print("====== MODEL ======")
print(f"Light Intensity Coefficient: {w[0]:.4f}")
print(f"Panel Temperature Coefficient: {w[1]:.4f}")
print(f"Voltage Coefficient: {w[2]:.4f}")
print(f"Current Coefficient: {w[3]:.4f}")
print(f"Intercept: {b:.4f}")

print("\n====== NEW SOLAR POWER ======")
print(f"Light Intensity: {new_input[0]} W/m^2")
print(f"Panel Temperature: {new_input[1]} °C")
print(f"Voltage: {new_input[2]} V")
print(f"Current: {new_input[3]} A")

print("\n======= METRICS =======")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r_squared:.2f}")

print("\n====== ACTUAL VS PREDICTED ======")
for actual, predicted in zip(Y, Y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

print("\n====== PREDICTED GENERATED POWER ======")
print(f"Predicted Generated Power: {new_output:.2f} Watts")

#plotting
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Loss (MSE)', color='blue')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

#plotting actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(range(len(Y)), Y, label='Actual', color='blue', alpha=0.6)
plt.scatter(range(len(Y_pred)), Y_pred, label='Predicted', color='orange',
    alpha=0.6)
plt.title('Actual vs Predicted Generated Power')
plt.xlabel('Sample Index')
plt.ylabel('Generated Power (Watts)')
plt.legend()
plt.show()

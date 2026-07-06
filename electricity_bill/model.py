import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("electricity_bill_dataset.csv")
X = df[["UnitsConsumed", "FamilyMembers", "Appliances", "AC_Hours"]].values
Y = df["ElectricityBill"].values

#only X
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Y_mean = np.mean(Y)
# Y_std = np.std(Y)
# Y = (Y - Y_mean) / Y_std

w = np.zeros(X.shape[1])
b = 0
n = len(X)
learning_rate = 0.01
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
    
    if(epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {mse:.4f}, Bias: {b:.4f}, Weights: {w}"
        )
Y_pred = np.dot(X, w) + b
new_input = np.array([
    350,    # Units Consumed
    4,      # Family Members
    10,     # Appliances
    5       # AC Hours
])
new_input_scaled = (new_input - X_mean) / X_std
new_output = np.dot(new_input_scaled, w) + b

#metrics
mse = np.mean((Y - Y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y - Y_pred))
ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_res = np.sum((Y - Y_pred) ** 2)
r_squared = 1 - (ss_res / ss_total)

#outputs
print("====== MODEL ======")
print(f"Units Consumed Coefficient: {w[0]:.4f}")
print(f"Family Members Coefficient: {w[1]:.4f}")
print(f"Appliances Coefficient: {w[2]:.4f}")
print(f"AC Hours Coefficient: {w[3]:.4f}")
print(f"Bias: {b:.4f}")

print("\n====== EVALUATION ======")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-Squared: {r_squared:.4f}")

print("\n====== NEW ELECTRICITY BILL PREDICTION ======")
print(f"Predicted Electricity Bill: {new_output:.4f}")
print(f"Units Consumed: {new_input[0]}")
print(f"Family Members: {new_input[1]}")
print(f"Appliances: {new_input[2]}")
print(f"AC Hours: {new_input[3]}")

print('\n====== ACTUAL VS PREDICTED ======')
for actual, predicted in zip(Y, Y_pred):
    print(f"Actual: {actual:.4f}, Predicted: {predicted:.4f}")

#plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), losses, color='blue')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.show()

#plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(range(len(Y)), Y, color='blue', label='Actual Electricity Bills')
plt.scatter(range(len(Y_pred)), Y_pred, color='red', label='Predicted Electricity Bills')
plt.title('Actual vs Predicted Electricity Bills')
plt.xlabel('Index')
plt.ylabel('Electricity Bill')
plt.legend()
plt.show()


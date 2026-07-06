import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("medical_insurance_dataset.csv")

print("====== ORIGINAL DATA ======")
print(df.head())

print("\n====== MISSING VALUES ======")
print(df.isnull().sum())

#fill missing bmi with mean
df["BMI"] = df["BMI"].fillna(df["BMI"].mean())

#fill missing insruance charges with mean
df["InsuranceCost"] = df["InsuranceCost"].fillna(df["InsuranceCost"].mean())

#convert categorical variables to numerical
df["Smoker"] = df["Smoker"].map({"Yes": 1, "No": 0})

# print("\n====== DATA AFTER PREPROCESSING ======")
# print(df.to_string())

X = df[["Age", "BMI","Children","Smoker"]].values
Y = df["InsuranceCost"].values

w = np.zeros(4)
b = 0
learning_rate = 0.0001
epochs = 5000
n = len(X)
losses = []

for epoch in range(epochs):
    Y_pred = np.dot(X, w) + b
    loss = (1/n) * np.sum((Y_pred - Y) ** 2) #-> mean squared error
    losses.append(loss)

    dw = (2/n) * np.dot(X.T, (Y_pred - Y))
    db = (2/n) * np.sum(Y_pred - Y)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Final prediction
Y_pred = np.dot(X, w) + b

# Metrics
mse = np.mean((Y - Y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y - Y_pred))

ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_residual = np.sum((Y - Y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

print("\n====== ACTUAL vs PREDICTED ======")

for actual, predicted in zip(Y, Y_pred):
    print(f"Actual: {actual:.2f}  Predicted: {predicted:.2f}")

print("\n========== MODEL ==========")
print(f"Age Coefficient: {w[0]:.4f}")
print(f"BMI Coefficient: {w[1]:.4f}")
print(f"Children Coefficient: {w[2]:.4f}")
print(f"Smoker Coefficient: {w[3]:.4f}")
print("Weights:", w)
print(f"Bias: {b:.4f}")

print("\n========== METRICS ==========")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)
print("R²  :", r2)

new_customer = np.array([35,28.5,2,1])#-> Age, BMI, Children, Smoker
predicted_cost = np.dot(new_customer, w) + b

print("\n========== NEW CUSTOMER PREDICTION ==========")
print("Age: ",new_customer[0])
print("BMI: ",new_customer[1])
print("Children: ",new_customer[2])
print("Smoker: ","Yes" if new_customer[3] == 1 else "No")
print(f"Predicted Insurance Cost: ${predicted_cost:.2f}")

# Loss Curve
plt.figure(figsize=(8,5))
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# # Actual vs Predicted
# plt.scatter(range(len(Y)), Y, color='blue', label='Actual Costs')
# plt.scatter(range(len(Y_pred)), Y_pred, color='red', label="Predicted Costs")
# plt.title('Actual vs Predicted Insurance Costs')
# plt.xlabel('Customer Index')
# plt.ylabel('Insurance Cost')
# plt.legend()
# plt.show()

plt.figure(figsize=(8,5))

plt.scatter(Y, Y_pred)

minimum = min(Y.min(), Y_pred.min())
maximum = max(Y.max(), Y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    'r--',
    label="Perfect Prediction"
)

# plt.xlabel("Actual Insurance Cost")
# plt.ylabel("Predicted Insurance Cost")
# plt.title("Actual vs Predicted")
# plt.legend()
# plt.grid(True)
# plt.show()

#plot with linear line
# plt.scatter(range(len(Y)), Y, color='blue', label='Actual Costs')
# plt.plot(range(len(Y_pred)), Y_pred, color='red', label='Predicted Costs', linewidth=2)
# plt.title('Actual vs Predicted Insurance Costs')
# plt.xlabel('Customer Index')
# plt.ylabel('Insurance Cost')
# plt.legend()
# plt.show()
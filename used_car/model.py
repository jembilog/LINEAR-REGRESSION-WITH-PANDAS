import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("used_car_price_dataset.csv")

print("====== ORIGINAL DATA ======")
print(df.head())

print("\n====== MISSING VALUES ======")
print(df.isnull().sum())

df = df.drop_duplicates()

df["Mileage"] = df["Mileage"].fillna(df["Mileage"].mean())
df["Price"] = df["Price"].fillna(df["Price"].mean())

df = df[(df["Mileage"] >= 0)]
df = df[(df["Horsepower"] > 50) & (df["Horsepower"] < 500)]
df = df[(df["EngineSize"] > 0.8) & (df["EngineSize"] < 5.0)]
df = df[(df["Price"] < 3000000)]

print("\n====== CLEANED DATA ======")
print(df)

X = df[["Age","Mileage","EngineSize","Horsepower"]].values
Y = df["Price"].values

w = np.zeros(4)
b = 0

learning_rate = 0.000000001
epochs = 10000
n = len(X)

losses = []

for epoch in range(epochs):

    Y_pred = np.dot(X,w) + b

    mse = np.mean((Y_pred - Y) ** 2)
    losses.append(mse)

    dw = (2/n) * np.dot(X.T,(Y_pred - Y))
    db = (2/n) * np.sum(Y_pred - Y)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss = {mse:.2f}")

Y_pred = np.dot(X,w) + b

mse = np.mean((Y - Y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y - Y_pred))

ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_residual = np.sum((Y - Y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

print("\n====== MODEL ======")
print(f"Age Coefficient        : {w[0]:.6f}")
print(f"Mileage Coefficient    : {w[1]:.6f}")
print(f"Engine Coefficient     : {w[2]:.6f}")
print(f"Horsepower Coefficient : {w[3]:.6f}")
print(f"Bias                  : {b:.6f}")

print("\n====== EQUATION ======")
print(f"Price = ({w[0]:.6f} × Age) + ({w[1]:.6f} × Mileage) + ({w[2]:.6f} × EngineSize) + ({w[3]:.6f} × Horsepower) + ({b:.6f})")

print("\n====== METRICS ======")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R²   : {r2:.4f}")

print("\n====== ACTUAL VS PREDICTED ======")
for actual, predicted in zip(Y, Y_pred):
    print(f"Actual: {actual:.2f}\tPredicted: {predicted:.2f}")

new_car = np.array([5,60000,1.8,140])

prediction = np.dot(new_car,w) + b

print("\n====== NEW PREDICTION ======")
print("Age        :",new_car[0],"years")
print("Mileage   :",new_car[1],"km")
print("Engine    :",new_car[2],"L")
print("Horsepower:",new_car[3],"HP")
print(f"Predicted Price: ₱{prediction:.2f}")

plt.figure(figsize=(8,5))
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(Y,Y_pred)

minimum = min(Y.min(),Y_pred.min())
maximum = max(Y.max(),Y_pred.max())

plt.plot([minimum,maximum],[minimum,maximum],'r--',label="Perfect Prediction")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()
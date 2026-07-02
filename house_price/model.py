import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("house_price_dataset.csv")

X = df[["Area","Bedrooms","Bathrooms","Age"]].values
Y = df[["Price"]].values

#scaling
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

# Y_mean = np.mean(Y)
# Y_std = np.std(Y)

X = (X - X_mean) / X_std
# Y = (Y - Y_mean) / Y_std

w = np.zeros((X.shape[1],1))
b = 0
n = len(X)

learning_rate = 0.01
epochs = 10000
losses = []

for epoch in range(epochs):
    Y_pred = np.dot(X, w) + b
    mse = np.mean((Y - Y_pred) ** 2)
    losses.append(mse)
    dw = (-2/n) * np.dot(X.T, (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if(epoch + 1) % 500 == 0:
        print(
            f"MSE:{mse:.4f} | "
            f"Bias: {b:.11f}"
        )

Y_pred =np.dot(X, w) + b

new_house = np.array([153, 4 , 2, 220000])
new_house_scaled = (new_house - X_mean) / X_std
prediction = np.dot(new_house_scaled, w) + b

#evaluation
mse = np.mean((Y - Y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y - Y_pred))
ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_residual = np.sum((Y - Y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

# print(r2)
# print(mse)
# print(rmse)
# print(mae)
# print(b)
# print(w)

print("========== MODEL ==========")
print(f"Area Coefficient        : {w[0].item():.4f}")
print(f"Bedrooms Coefficient    : {w[1].item():.4f}")
print(f"Bathrooms Coefficient   : {w[2].item():.4f}")
print(f"Age Coefficient         : {w[3].item():.4f}")
print(f"Bias                    : {b:.9f}")
print("\n========== NEW HOUSE ==========")
print(f"Area       : {new_house[0]} m^2")
print(f"Bedrooms : {new_house[1]}")
print(f"Bathrooms   : {new_house[2]}")
print(f"Age    : {int(new_house[3])}")
print(f"Predicted House Price: {prediction.item():.4f}")
print("\n========== METRICS ==========")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
print("\n========== ACTUAL VS PREDICTED ==========")

for actual, predicted in zip(Y, Y_pred):
    print(
        f"Actual: {actual.item():.2f} PHP\tPredicted: {predicted.item():.2f} PHP"
    )

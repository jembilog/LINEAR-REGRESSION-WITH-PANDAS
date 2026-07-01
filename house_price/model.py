import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("house_price_dataset.csv")

X = df[["Area","Bedrooms","Bathrooms","Age"]].values
Y = df[["Price"]].values

#scaling
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

Y_mean = np.mean(Y)
Y_std = np.std(Y)

X = (X - X_mean) / X_std
Y = (Y - Y_mean) / Y_std

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
            f"MSE:{mse:.4f}"
        )

Y_pred =np.dot(X, w) + b

new_car = np.array([153, 4 , 2, 220000])
new_car_scaled = (new_car - X_mean) / X_std
prediction = np.dot(new_car_scaled, w) + b

#evaluation
mse = np.mean((Y - Y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y - Y_pred))
ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_residual = np.sum((Y - Y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

print(r2)
print(mse)
print(rmse)
print(mae)
print(b)
print(w)
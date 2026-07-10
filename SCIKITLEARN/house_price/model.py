import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("house_price.csv")

X = df[["Area","Bedrooms","Bathrooms","Age"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print("\nModel Evaluation")
print("MAE :", mean_absolute_error(y_test,predictions))
print("MSE :",mean_squared_error(y_test,predictions))
print("R2 : ",r2_score(y_test,predictions))

print("\nIntercept:")
print(model.intercept_)
print("\nCoefficients:")

coefficients = pd.DataFrame({
    "Feature":X.columns,
    "Coefficient": model.coef_
})
print(coefficients)

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions.round(2)
})
print("\nActual vs Predicted")
print(comparison)

new_house = pd.DataFrame({
    "Area": [125],
    "Bedrooms": [4],
    "Bathrooms": [3],
    "Age": [3]
})

predicted_price = model.predict(new_house)
print(new_house)
print("\nPredicted Price: $", round(predicted_price[0], 2))

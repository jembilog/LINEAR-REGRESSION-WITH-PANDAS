import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student_exam_dataset.csv")

X = df[["StudyHours", "Attendance", "SleepHours", "AssignmentsCompleted"]].values
Y = df["ExamScore"].values

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

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

    print(
        f"Epoch {epoch+1}/{epochs}, Loss: {mse:.4f}, Bias: {b:.4f}, Weights: {w}"
    )

Y_pred = np.dot(X, w) + b
new_input = np.array([
    5,      # Study Hours
    90,     # Attendance (%)
    7,      # Sleep Hours
    10      # Assignments Completed
])
new_input_scaled = (new_input - X_mean) / X_std
new_output = np.dot(new_input_scaled, w) + b
# print(f"Predicted Exam Score: {new_output}")
mse = np.mean((Y - Y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y - Y_pred))
ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_res = np.sum((Y - Y_pred) ** 2)
r_squared = 1 - (ss_res / ss_total)

#output
print("====== MODEL ======")
print(f"Study Hours Coefficient: {w[0]:.4f}")
print(f"Attendance Coefficient: {w[1]:.4f}")
print(f"Sleep Hours Coefficient: {w[2]:.4f}")
print(f"Assignments Completed Coefficient: {w[3]:.4f}")
print(f"Bias: {b:.4f}")

print("\n====== EVALUATION ======")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared: {r_squared:.4f}")

print("\n====== NEW STUDENT EXAM SCORE ======")
print(f"Predicted Exam Score: {new_output:.4f}")
print(f"Study Hours: {new_input[0]}")
print(f"Attendance: {new_input[1]}%")
print(f"Sleep Hours: {new_input[2]}")
print(f"Assignments Completed: {new_input[3]}")

print("\n======= Actual vs Predicted Exam Scores =======")
for actual, predicted in zip(Y, Y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")
plt.scatter(range(len(Y)), Y, color='blue', label='Actual Exam Scores')
plt.scatter(range(len(Y_pred)), Y_pred, color='red', label='Predicted Exam Scores')
plt.title('Actual vs Predicted Exam Scores')
plt.xlabel('Student Index')
plt.ylabel('Exam Score')
plt.legend()
plt.show()
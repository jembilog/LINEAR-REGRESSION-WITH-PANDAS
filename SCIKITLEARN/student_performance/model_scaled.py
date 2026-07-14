import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = pd.read_csv("student_performance.csv")
fill_col = ["StudyHours" , "Attendance", "SleepHours", "AssignmentsCompleted", "PreviousGrade",]
for col in fill_col:
    df[col] = df[col].fillna(df[col].median()).abs()
df["InternetAccess"] = df["InternetAccess"].fillna(df["InternetAccess"].mode()[0]).str.strip().str.lower()
le = LabelEncoder()

df["InternetAccess"] = le.fit_transform(df["InternetAccess"])
X = df[["StudyHours", "Attendance", "SleepHours", "AssignmentsCompleted", "PreviousGrade", "InternetAccess"]]
y = df["FinalExamScore"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()

scaler = StandardScaler()
numerical_columns = [
    "StudyHours",
    "Attendance",
    "SleepHours",
    "AssignmentsCompleted",
    "PreviousGrade"
]
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nMETRICS")
print(mean_absolute_error(y_test,predictions))
print(mean_squared_error(y_test, predictions))
print(r2_score(y_test, predictions))

coef = pd.DataFrame({ 
    "Feature" : X.columns,
    "Weights" : model.coef_
})
print("\nCoefficients")
print(coef)

comparison = pd.DataFrame({
    "Actual" : y_test.values,
    "Predicted" : predictions.round(2)
})
print(comparison)
print(df)
new_student = pd.DataFrame({
    "StudyHours" : [5.0],
    "Attendance" : [67],
    "SleepHours" : [6],
    "AssignmentsCompleted"  : [10],
    "PreviousGrade" : [87],
    "InternetAccess" : [0]
})
new_student[numerical_columns] = scaler.transform(
    new_student[numerical_columns]
)
new_prediction = model.predict(new_student)
print("FinalExamScore",new_prediction)

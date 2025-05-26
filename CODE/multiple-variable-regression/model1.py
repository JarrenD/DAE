import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("./CSV/gorilla_tug_of_war.csv")

df = pd.get_dummies(df, columns=['SUS'])
df = pd.get_dummies(df, columns=['GND'],drop_first=True)

y = df["HMNS"]
selected_features = ['WHT', 'AGE', 'DSI']
X = df[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Test R²:", round(test_r2, 3))
print("Test MSE:", round(test_mse, 3))
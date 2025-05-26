import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

df = pd.read_csv("./CSV/gorilla_tug_of_war.csv")
df = pd.get_dummies(df, columns=['SUS'])
df = pd.get_dummies(df, columns=['GND'], drop_first=True)

selected_features = ['WHT', 'DSI', "SUS_Grauer's", 'GND_Male']
X = df[selected_features]
y = df["HMNS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = X_train.copy()
train_df["HMNS"] = y_train

numerical_cols = ['WHT', 'DSI', 'HMNS']
z_scores = train_df[numerical_cols].apply(zscore)

train_df_no_outliers = train_df[(z_scores.abs() < 3).all(axis=1)]

X_train_clean = train_df_no_outliers.drop(columns=["HMNS"])
y_train_clean = train_df_no_outliers["HMNS"]

scaler = MinMaxScaler()
X_train_clean_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_clean_scaled, y_train_clean)

y_train_pred = model.predict(X_train_clean_scaled)
y_test_pred = model.predict(X_test_scaled)

train_r2 = r2_score(y_train_clean, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train_clean, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Train R²:", round(train_r2, 3))
print("Test R²:", round(test_r2, 3))
print("Train MSE:", round(train_mse, 3))
print("Test MSE:", round(test_mse, 3))
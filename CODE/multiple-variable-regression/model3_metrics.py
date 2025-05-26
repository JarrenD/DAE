import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore

df = pd.read_csv("./CSV/gorilla_tug_of_war.csv")
df = pd.get_dummies(df, columns=['SUS'])
df = pd.get_dummies(df, columns=['GND'], drop_first=True)

selected_features = ['WHT', 'FRC', 'AGE', 'DSI', 'SUS_Cross River', 'SUS_Grauer\'s', 'SUS_Mountain', 'SUS_Western Lowland', 'GND_Male']
X = df[selected_features]
y = df["HMNS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = X_train.copy()
train_df["HMNS"] = y_train
numerical_cols = ['WHT', 'FRC', 'AGE', 'DSI', 'HMNS']
z_scores = train_df[numerical_cols].apply(zscore)
train_df_no_outliers = train_df[(z_scores.abs() < 3).all(axis=1)]
X_train_clean = train_df_no_outliers.drop(columns=["HMNS"])
y_train_clean = train_df_no_outliers["HMNS"]

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print("\n🔍 Running Grid Search for Ridge and Lasso...")

lasso_params = {'alpha': [0.01, 0.1, 1, 10, 100]}

lasso = GridSearchCV(Lasso(max_iter=10000), lasso_params, scoring='r2', cv=5)
lasso.fit(X_train_poly, y_train_clean)

y_train_pred = lasso.predict(X_train_poly)
y_test_pred = lasso.predict(X_test_poly)

train_r2 = r2_score(y_train_clean, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train_clean, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Train R²:", round(train_r2, 3))
print("Test R²:", round(test_r2, 3))
print("Train MSE:", round(train_mse, 3))
print("Test MSE:", round(test_mse, 3))
print(f"Best alpha: {lasso.best_params_['alpha']}")

train_residuals = y_train_clean - y_train_pred
test_residuals = y_test - y_test_pred

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_train_pred, y=train_residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Train Residuals vs. Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test_pred, y=test_residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Test Residuals vs. Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_test_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Actual vs. Predicted (Test Set)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.show()
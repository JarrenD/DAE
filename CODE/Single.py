import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('../CSV/gorilla_tug_of_war.csv')
data_clean = data.dropna()
Weight = data_clean[['WHT']]
Age = data_clean[['AGE']]
DSI = data_clean[['DSI']]
Species = pd.get_dummies(data_clean, columns=['SUS'])
SUS_features = Species[['SUS_Western Lowland', 'SUS_Mountain', 'SUS_Grauer\'s', 'SUS_Cross River']]

Target = data_clean['HMNS']

#----Global Target-----
Target_train = Target[:375]
Target_test = Target[375:]

#-----WEIGHT-----

Weight_train = Weight[:375] # train data according to weight
Weight_test = Weight[375:]  # used for testing, compared with Target_test

model = LinearRegression()
model.fit(Weight_train,Target_train)

prediction_Weight = model.predict(Weight_test)                # model predicts data using unseen weight
R2_value_weight = r2_score(Target_test,prediction_Weight)     # R2 value
print("R² on test set for Weight:", R2_value_weight)

plt.scatter(Weight_test, Target_test, color='blue', label='Actual')
plt.plot(Weight_test, prediction_Weight, color='red', linewidth=2, label='Predicted')

plt.xlabel("Weight")
plt.ylabel("Humans Defeated (HMNS)")
plt.title("Linear Regression: Weight vs Humans Defeated")
plt.legend()
plt.savefig('../GRAPHS/WEIGHT_vs_hmns.png')
plt.show()

#-----AGE-----

Age_train = Age[:375] # train data according to age
Age_test = Age[375:]  # used for testing, compared with Target_test

model = LinearRegression()
model.fit(Age_train,Target_train)

prediction_Age = model.predict(Age_test)                    # model predicts data using unseen age
R2_value_Age = r2_score(Target_test,prediction_Age)         # R2 value
print("R² on test set for Age:", R2_value_Age)

plt.scatter(Age_test, Target_test, color='blue', label='Actual')
plt.plot(Age_test, prediction_Age, color='red', linewidth=2, label='Predicted')

plt.xlabel("Age")
plt.ylabel("Humans Defeated (HMNS)")
plt.title("Linear Regression: Age vs Humans Defeated")
plt.legend()
plt.savefig('../GRAPHS/AGE_vs_hmns.png')
plt.show()

#-----DSI-----

DSI_train = DSI[:375] # train data according to dsi
DSI_test = DSI[375:]  # used for testing, compared with Target_test

model = LinearRegression()
model.fit(DSI_train,Target_train)

prediction_DSI = model.predict(DSI_test)                    # model predicts data using unseen dsi
R2_value_DSI = r2_score(Target_test,prediction_DSI)         # R2 value
print("R² on test set for DSI:", R2_value_DSI)

plt.scatter(DSI_test, Target_test, color='blue', label='Actual')
plt.plot(DSI_test, prediction_DSI, color='red', linewidth=2, label='Predicted')

plt.xlabel("DSI")
plt.ylabel("Humans Defeated (HMNS)")
plt.title("Linear Regression: DSI vs Humans Defeated")
plt.legend()
plt.savefig('../GRAPHS/DSI_vs_hmns.png')
plt.show()

#-----SUS-----

SUS_train = SUS_features[:375] # train data according to each species
SUS_test = SUS_features[375:]  # used for testing, compared with Target_test

model = LinearRegression()
model.fit(SUS_train,Target_train)

prediction_SUS = model.predict(SUS_test)                    # model predicts data using unseen species data
R2_value_SUS = r2_score(Target_test,prediction_SUS)         # R2 value
print("R² on test set for Species:", R2_value_SUS)

# More of averaged data
sns.barplot(x='SUS', y='HMNS', data=data_clean, errorbar=None)
plt.title('Average Humans Defeated per Gorilla Sub-Species')
plt.xlabel('Sub-Species')
plt.ylabel('Avg Humans Defeated (HMNS)')
plt.savefig('../GRAPHS/SUS_barplot.png')
plt.show()

# How the data is distributed across the differ species
sns.boxplot(x='SUS', y='HMNS', data=data_clean)
plt.title('Distribution of HMNS per Sub-Species')
plt.xlabel('Sub-Species')
plt.ylabel('Humans Defeated')
plt.savefig('../GRAPHS/SUS_boxplot.png')
plt.show()
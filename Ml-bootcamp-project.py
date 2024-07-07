# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:58:40 2024

@author: Yunus
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
data = pd.read_csv("C:/Users/yunus/Downloads/insurance.csv")

# Data control
print(data.head())
print(data.describe().T)

# Examine the distribution of BMI
sns.histplot(data['bmi'], bins=30)
plt.title('BMI Distribution')
plt.show()

# Examine the relationship between "smoker" and "expenses"
sns.boxplot(x='smoker', y='expenses', data=data)
plt.title('Smoker vs Expenses')
plt.show()

# Examine the relationship between "smoker" and "region"
sns.countplot(x='region', hue='smoker', data=data)
plt.title('Smoker vs Region')
plt.show()

# Examine the relationship between "bmi" and "sex"
sns.boxplot(x='sex', y='bmi', data=data)
plt.title('BMI vs Sex')
plt.show()

# Find the "region" with the most "children"
region_with_most_children = data.groupby('region')['children'].sum().idxmax()
print(f'Region with most children: {region_with_most_children}')

# Examine the relationship between "age" and "bmi"
sns.scatterplot(x='age', y='bmi', data=data)
plt.title('Age vs BMI')
plt.show()

# Examine the relationship between "bmi" and "children"
sns.boxplot(x='children', y='bmi', data=data)
plt.title('BMI vs Children')
plt.show()

# Check for outliers in "bmi"
sns.boxplot(x=data['bmi'])
plt.title('BMI Outliers')
plt.show()

# Examine the relationship between "bmi" and "expenses"
sns.scatterplot(x='bmi', y='expenses', data=data)
plt.title('BMI vs Expenses')
plt.show()

# Examine the relationship between "region", "smoker", and "bmi" using a bar plot
sns.barplot(x='region', y='bmi', hue='smoker', data=data)
plt.title('Region, Smoker, and BMI')
plt.show()

# Creating model
cat_col = ['smoker', 'region', 'sex']
num_col = [i for i in data.columns if i not in cat_col]
print(num_col)

# One-hot encoding
one_hot = pd.get_dummies(data[cat_col])
insur_procsd_data = pd.concat([data[num_col], one_hot], axis=1)
print(insur_procsd_data.head(10))

# Label encoding
insr_procsd_data_label = data.copy()
label_encoder = LabelEncoder()
for col in cat_col:
    insr_procsd_data_label[col] = label_encoder.fit_transform(insr_procsd_data_label[col])
print(insr_procsd_data_label.head(10))

# Using one hot encoding
X = insur_procsd_data.drop(columns='expenses')
y = data['expenses']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1234)

# Model 1 - Linear Regression
model = LinearRegression()
model.fit(train_X, train_y)

# Print Model intercept and coefficient
print("Model intercept", model.intercept_, "Model coefficient", model.coef_)

cdf = pd.DataFrame(data=model.coef_.T, index=X.columns, columns=["Coefficients"])
print(cdf)

# Print various metrics
print("Predicting the train data")
train_predict = model.predict(train_X)
print("Predicting the test data")
test_predict = model.predict(test_X)

print("MAE")
print("Train : ", mean_absolute_error(train_y, train_predict))
print("Test  : ", mean_absolute_error(test_y, test_predict))
print("====================================")
print("MSE")
print("Train : ", mean_squared_error(train_y, train_predict))
print("Test  : ", mean_squared_error(test_y, test_predict))
print("====================================")
print("RMSE")
print("Train : ", np.sqrt(mean_squared_error(train_y, train_predict)))
print("Test  : ", np.sqrt(mean_squared_error(test_y, test_predict)))
print("====================================")
print("R^2")
print("Train : ", r2_score(train_y, train_predict))
print("Test  : ", r2_score(test_y, test_predict))
print("MAPE")
print("Train : ", np.mean(np.abs((train_y - train_predict) / train_y)) * 100)
print("Test  : ", np.mean(np.abs((test_y - test_predict) / test_y)) * 100)

# Plot actual vs predicted value
plt.figure(figsize=(10, 7))
plt.title("Actual vs. Predicted Expenses", fontsize=25)
plt.xlabel("Actual Expenses", fontsize=18)
plt.ylabel("Predicted Expenses", fontsize=18)
plt.scatter(x=test_y, y=test_predict)
plt.show()

# Model 2 - RandomForest
model2 = RandomForestRegressor()
model2.fit(train_X, train_y)

# Print various metrics
print("Predicting the train data")
train_predict2 = model2.predict(train_X)
print("Predicting the test data")
test_predict2 = model2.predict(test_X)

print("Model 2 MAE")
print("Train : ", mean_absolute_error(train_y, train_predict2))
print("Test  : ", mean_absolute_error(test_y, test_predict2))
print("====================================")
print("Model 2 MSE")
print("Train : ", mean_squared_error(train_y, train_predict2))
print("Test  : ", mean_squared_error(test_y, test_predict2))
print("====================================")
print("Model 2 RMSE")
print("Train : ", np.sqrt(mean_squared_error(train_y, train_predict2)))
print("Test  : ", np.sqrt(mean_squared_error(test_y, test_predict2)))
print("====================================")
print("Model 2 R^2")
print("Train : ", r2_score(train_y, train_predict2))
print("Test  : ", r2_score(test_y, test_predict2))
print("Model 2 MAPE")
print("Train : ", np.mean(np.abs((train_y - train_predict2) / train_y)) * 100)
print("Test  : ", np.mean(np.abs((test_y - test_predict2) / test_y)) * 100)

# Plot actual vs predicted value
plt.figure(figsize=(10, 7))
plt.title("Actual vs. Predicted Expenses", fontsize=25)
plt.xlabel("Actual Expenses", fontsize=18)
plt.ylabel("Predicted Expenses", fontsize=18)
plt.scatter(x=test_y, y=test_predict2)
plt.show()

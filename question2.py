import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = pd.read_csv('linear_regression_dataset.csv')

data.columns = data.columns.str.strip()
X = data[['Height']]  
y = data['Weight']   


model = LinearRegression()
model.fit(X, y)
slope_inbuilt = model.coef_[0]
intercept_inbuilt = model.intercept_


x_mean = np.mean(X['Height'])
y_mean = np.mean(y)
numerator = np.sum((X['Height'] - x_mean) * (y - y_mean))
denominator = np.sum((X['Height'] - x_mean) ** 2)
slope_manual = numerator / denominator
intercept_manual = y_mean - (slope_manual * x_mean)


plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Inbuilt Regression Line')
plt.plot(X, slope_manual * X['Height'] + intercept_manual, color='green', linestyle='--', label='Manual Regression Line')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.show()


print(f"Inbuilt Model - Slope: {slope_inbuilt}, Intercept: {intercept_inbuilt}")
print(f"Manual Calculation - Slope: {slope_manual}, Intercept: {intercept_manual}")
print("Columns in dataset:", data.columns)

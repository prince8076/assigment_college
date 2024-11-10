# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# file_path = pd.read_csv('linear_regression_dataset.csv')


# data = pd.read_csv(file_path)
# data.fillna(data.mean(numeric_only=True), inplace=True)
# data = data[['Price', 'Horsepower']].dropna()
# X = data['Horsepower'].values.reshape(-1, 1)
# y = data['Price'].values 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(X_train, y_train)
# print(f"Coefficient: {model.coef_[0]:.4f}")
# print(f"Intercept: {model.intercept_:.4f}")
# y_pred_sklearn = model.predict(X_test)
# mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
# r2_sklearn = r2_score(y_test, y_pred_sklearn)

# print(f"Mean Squared Error (sklearn): {mse_sklearn:.4f}")
# print(f"R2 Score (sklearn): {r2_sklearn:.4f}")

# # plot
# plt.figure(figsize=(8, 5))
# plt.scatter(X_test, y_test, color='blue', label='Actual Data')
# plt.plot(X_test, y_pred_sklearn, color='red', linewidth=2, label='Regression Line (sklearn)')
# plt.xlabel('Horsepower')
# plt.ylabel('Price')
# plt.title('Linear Regression using sklearn')
# plt.legend()
# plt.grid(True)
# plt.show()


# def manual_linear_regression(X, y):
#     X_mean = np.mean(X)
#     y_mean = np.mean(y)
#     numerator = np.sum((X - X_mean) * (y - y_mean))
#     denominator = np.sum((X - X_mean) ** 2)
#     m = numerator / denominator
#     c = y_mean - m * X_mean
#     return m, c

# m_manual, c_manual = manual_linear_regression(X_train.flatten(), y_train)
# print(f"Manual Coefficient (slope): {m_manual:.4f}")
# print(f"Manual Intercept: {c_manual:.4f}")
# y_pred_manual = m_manual * X_test.flatten() + c_manual
# mse_manual = mean_squared_error(y_test, y_pred_manual)
# r2_manual = r2_score(y_test, y_pred_manual)

# print(f"Mean Squared Error (manual): {mse_manual:.4f}")
# print(f"R2 Score (manual): {r2_manual:.4f}")
# plt.figure(figsize=(8, 5))
# plt.scatter(X_test, y_test, color='blue', label='Actual Data')
# plt.plot(X_test, y_pred_manual, color='green', linewidth=2, label='Regression Line (Manual)')
# plt.xlabel('Horsepower')
# plt.ylabel('Price')
# plt.title('Manual Linear Regression')
# plt.legend()
# plt.grid(True)
# plt.show()








import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('linear_regression_dataset.csv')

# Display column names
print("Columns in dataset:", data.columns)

# Remove any extra spaces from column names (optional, to avoid issues with trailing spaces)
data.columns = data.columns.str.strip()

# Define the variables (assuming 'Height' as the independent variable and 'Weight' as the dependent variable)
X = data[['Height']]  # Using the 'Height' column
y = data['Weight']    # Using the 'Weight' column

# Part 1: Inbuilt Linear Regression
model = LinearRegression()
model.fit(X, y)
slope_inbuilt = model.coef_[0]
intercept_inbuilt = model.intercept_

# Part 2: Manual Linear Regression Calculation
x_mean = np.mean(X['Height'])
y_mean = np.mean(y)
numerator = np.sum((X['Height'] - x_mean) * (y - y_mean))
denominator = np.sum((X['Height'] - x_mean) ** 2)
slope_manual = numerator / denominator
intercept_manual = y_mean - (slope_manual * x_mean)

# Part 3: Plotting both regression lines on the same scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Inbuilt Regression Line')
plt.plot(X, slope_manual * X['Height'] + intercept_manual, color='green', linestyle='--', label='Manual Regression Line')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.show()

# Print results for both models
print(f"Inbuilt Model - Slope: {slope_inbuilt}, Intercept: {intercept_inbuilt}")
print(f"Manual Calculation - Slope: {slope_manual}, Intercept: {intercept_manual}")

# Display column names again to verify
print("Columns in dataset:", data.columns)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("iris.csv")

# 1. Visualize the distribution of each feature and the class distribution
plt.figure(figsize=(12, 8))
for i, column in enumerate(data.columns[:-1], 1):
    plt.subplot(2, 2, i)
    sns.histplot(data[column], kde=True)
    plt.title(f"Distribution of {column}")
plt.tight_layout()
plt.show()

# Class distribution - Updated to use 'variety' instead of 'species'
sns.countplot(data=data, x="variety")
plt.title("Class Distribution of Iris Species")
plt.show()

# 2. Encode the categorical target variable (variety) into numerical values
le = LabelEncoder()
data['variety'] = le.fit_transform(data['variety'])

# 3. Split the dataset into training and testing sets (80% train, 20% test)
X = data.drop('variety', axis=1)
y = data['variety']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Decision Tree Model
# i. Build a decision tree classifier using the training set
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# ii. Visualize the resulting decision tree
plt.figure(figsize=(15, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True)
plt.show()

# iii. Make predictions on the testing set and evaluate the model's performance
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# 5. Random Forest Model
# i. Build a random forest classifier using the training set
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# ii. Make predictions on the testing set and evaluate the model's performance
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

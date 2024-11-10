import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
file_path = 'logistic_regression_dataset.csv'
data = pd.read_csv(file_path)


le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])


X = data[['Gender', 'Age', 'EstimatedSalary']]
y = data['Purchased']


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = LogisticRegression(random_state=0, max_iter=200)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
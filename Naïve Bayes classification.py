# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load Iris dataset (CSV or fallback)
try:
    df = pd.read_csv('Iris.csv')
    if 'Id' in df.columns:
        df.drop('Id', axis=1, inplace=True)
except:
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target

# Split data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy & Error Rate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Error Rate:", 1 - accuracy)

# TP, FP, FN, TN for class 0 (example)
TP = cm[0][0]
FP = sum(cm[:,0]) - TP
FN = sum(cm[0,:]) - TP
TN = cm.sum() - (TP + FP + FN)

print("\nFor Class 0:")
print("TP:", TP, "FP:", FP, "FN:", FN, "TN:", TN)

# Precision & Recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("Precision:", precision)
print("Recall:", recall)
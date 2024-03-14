# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv("/creditcard.csv")

# Basic dataset exploration
print(df.head())
print(df.shape)
print(df.isnull().sum())

# Distribution of Normal vs Fraud transactions
fraud_distribution = pd.value_counts(df['Class'], sort=True)
fraud_distribution.plot(kind='bar', rot=0, color='red')
plt.title("Transaction Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
labels = ['Normal', 'Fraud']  # Defining labels for the classes
plt.xticks(range(2), labels)
plt.show()

# Separating the data into fraud and normal transactions
fraud_transactions = df[df['Class'] == 1]
normal_transactions = df[df['Class'] == 0]

print(fraud_transactions.shape)
print(normal_transactions.shape)

# Statistical summary of amounts in fraud and normal transactions
print(fraud_transactions['Amount'].describe())
print(normal_transactions['Amount'].describe())

# Plotting the distribution of transaction amounts by class
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.suptitle('Transaction Amount per Class')
bins = 70

ax1.hist(fraud_transactions['Amount'], bins=bins)
ax1.set_title('Fraud')

ax2.hist(normal_transactions['Amount'], bins=bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(30, 30))
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Preparing data for model training
columns = [c for c in df.columns if c != "Class"]
X = df[columns]
y = df["Class"]

print(X.shape)
print(y.shape)

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, max_samples=len(X_train), random_state=42)
iso_forest.fit(X_train, y_train)
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso[y_pred_iso == 1] = 0
y_pred_iso[y_pred_iso == -1] = 1

# Evaluation of Isolation Forest model
print("Isolation Forest Accuracy:", accuracy_score(y_test, y_pred_iso))
print("Classification Report:\n", classification_report(y_test, y_pred_iso))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_iso))
print("Isolation Forest Errors:", (y_pred_iso != y_test).sum())

# One-Class SVM model
svm_model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_pred_svm[y_pred_svm == 1] = 0
y_pred_svm[y_pred_svm == -1] = 1

# Evaluation of SVM model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("SVM Errors:", (y_pred_svm != y_test).sum())

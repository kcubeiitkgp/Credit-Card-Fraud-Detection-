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

# Ensure that you have already installed pycaret before running this snippet.
# If not, you can install it using pip in your environment.
# !pip install pycaret

# Import the classification module from pycaret
from pycaret.classification import setup, compare_models, create_model, tune_model, predict_model

# Load the dataset
try:
    df = pd.read_csv("creditcard.csv")
except FileNotFoundError:
    print("The file was not found. Please check the path and try again.")
    # Add any additional error handling as necessary

# Display the first few rows of the dataframe to ensure it's loaded correctly
print(df.head())

# Set up the environment in pycaret for classification
# Specify the dataframe and the target column
model_setup = setup(data=df, target='Class', silent=True, session_id=123)

# Compare different models to find the best performing one
best_model = compare_models()

# Create a model based on Random Forest classifier
random_forest = create_model('rf')

# Display the Random Forest model
print(random_forest)

# Tune the Random Forest model to optimize its performance
tuned_random_forest = tune_model(random_forest)

# Display the tuned model's parameters
print(tuned_random_forest)


 perform predictions on the hold-out set
pred_holdout = predict_model(tuned_random_forest, data=x_test)
print(pred_holdout)

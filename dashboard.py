import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv('Phising_dataset.csv', encoding='windows-1254')

# Drop index column
data = data.drop(['URL'], axis=1)

# Prepare data for modeling
X = data.drop(['Result'], axis=1)
y = data['Result']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create Naive Bayes model and train it
clf = GaussianNB()
clf.fit(X_train, y_train)

# Predict target values
y_train_nb = clf.predict(X_train)
y_test_nb = clf.predict(X_test)

# Compute confusion matrix
cm = metrics.confusion_matrix(y_test, y_test_nb)

# Streamlit app
st.title("Phishing URL Analysis Dashboard")

# Show dataset
st.subheader("Dataset")
st.write(data.head())

# Phishing count pie chart
st.subheader("Phishing URL Count")
fig1, ax1 = plt.subplots(figsize=(2, 2))  # Adjusted figure size
data['Result'].value_counts().plot(kind='pie', autopct='%1.2f%%', ax=ax1)
plt.title("Phishing URL Count")
st.pyplot(fig1)

# Confusion matrix heatmap
st.subheader("Confusion Matrix for Naive Bayes Model")
fig2, ax2 = plt.subplots(figsize=(3, 2))  # Adjusted figure size
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted -1", "Predicted 1"], yticklabels=["True -1", "True 1"], ax=ax2)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
st.pyplot(fig2)

# Display model performance metrics
acc_test_nb = metrics.accuracy_score(y_test, y_test_nb)
f1_score_test_nb = metrics.f1_score(y_test, y_test_nb)
recall_test_nb = metrics.recall_score(y_test, y_test_nb)
precision_test_nb = metrics.precision_score(y_test, y_test_nb)

st.subheader("Model Performance Metrics")
st.write(f"Accuracy on test data: {acc_test_nb:.3f}")
st.write(f"F1-score on test data: {f1_score_test_nb:.3f}")
st.write(f"Recall on test data: {recall_test_nb:.3f}")
st.write(f"Precision on test data: {precision_test_nb:.3f}")

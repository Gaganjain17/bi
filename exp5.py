# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("try.csv")

# Perform one-hot encoding on categorical variables
data = pd.get_dummies(data)

# Splitting the dataset into features (X) and target variable (y)
X = data.drop('Decision_Yes', axis=1) # Adjust column name if needed
y = data['Decision_Yes'] # Adjust column name if needed

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Decision Tree classifier with limited depth
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# Training the Decision Tree classifier
clf.fit(X_train, y_train)

# Making predictions on the testing set
y_pred = clf.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plotting the decision tree with proper size and structure
plt.figure(figsize=(15,10))
plot_tree(clf, feature_names=X.columns, class_names=['No Decision', 'Decision'], filled=True, fontsize=10)
plt.show()


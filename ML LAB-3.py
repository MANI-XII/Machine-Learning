#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np

# Load the dataset using the provided file path
file_path = "C:/Users/year3/Downloads/dataset.xlsx"
data = pd.read_excel("C:/Users/year3/Downloads/dataset.xlsx")

# Separate the features and labels
features = data.iloc[:, :-1]
labels = data['LABEL']

# Calculate the intraclass spread (standard deviation within each class)
intraclass_spread = features.groupby(labels).std().mean()

# Calculate the interclass spread (spread between the means of different classes)
class_means = features.groupby(labels).mean()
interclass_spread = class_means.std()

# Output the results
print("Intraclass Spread:")
print(intraclass_spread)

print("\nInterclass Spread:")
print(interclass_spread)


# In[14]:


import pandas as pd
import numpy as np

# Load the dataset
file_path = "C:/Users/year3/Downloads/dataset.xlsx"
data = pd.read_excel("C:/Users/year3/Downloads/dataset.xlsx")

# Separate the features and labels
features = data.iloc[:, :-1].values
labels = data['LABEL'].values

# Get the unique classes
classes = np.unique(labels)

# Initialize dictionaries to store centroids and spreads
centroids = {}
spreads = {}

# Calculate the mean (centroid) and spread for each class
for cls in classes:
    class_features = features[labels == cls]
    
    # Calculate the mean (centroid) for the class
    centroids[cls] = np.mean(class_features, axis=0)
    
    # Calculate the spread (standard deviation) for the class
    spreads[cls] = np.std(class_features, axis=0)

# Calculate the distance between mean vectors (centroids) of each pair of classes
distances = {}
for i, cls1 in enumerate(classes):
    for cls2 in classes[i+1:]:
        distance = np.linalg.norm(centroids[cls1] - centroids[cls2])
        distances[(cls1, cls2)] = distance

# Output the results
print("Class Centroids (Means):")
for cls, centroid in centroids.items():
    print(f"Class {cls}: {centroid}")

print("\nClass Spreads (Standard Deviations):")
for cls, spread in spreads.items():
    print(f"Class {cls}: {spread}")

print("\nDistances Between Class Centroids:")
for pair, distance in distances.items():
    print(f"Distance between class {pair[0]} and class {pair[1]}: {distance}")


# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:/Users/year3/Downloads/dataset.xlsx"
data = pd.read_excel("C:/Users/year3/Downloads/dataset.xlsx")

# Select a feature (e.g., the first feature/column)
selected_feature = data.iloc[:, 0]  # Change the index to select a different feature

# Calculate the mean and variance of the selected feature
mean = np.mean(selected_feature)
variance = np.var(selected_feature)

# Print the mean and variance
print(f"Mean of the selected feature: {mean}")
print(f"Variance of the selected feature: {variance}")

# Generate and plot the histogram
plt.hist(selected_feature, bins=30, edgecolor='black', alpha=0.7)  # You can adjust the number of bins as needed
plt.title('Histogram of Selected Feature')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.show()


# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:/Users/year3/Downloads/dataset.xlsx"
data = pd.read_excel("C:/Users/year3/Downloads/dataset.xlsx")

# Select two feature vectors (e.g., the first two rows)
vector1 = data.iloc[0, :-1].values  # Exclude the label column
vector2 = data.iloc[1, :-1].values  # Exclude the label column

# Function to calculate Minkowski distance
def minkowski_distance(vec1, vec2, r):
    return np.sum(np.abs(vec1 - vec2) ** r) ** (1/r)

# Calculate Minkowski distance for r from 1 to 10
r_values = np.arange(1, 11)
distances = [minkowski_distance(vector1, vector2, r) for r in r_values]

# Plot the Minkowski distance as a function of r
plt.plot(r_values, distances, marker='o')
plt.title('Minkowski Distance between Two Feature Vectors')
plt.xlabel('r')
plt.ylabel('Distance')
plt.xticks(r_values)  # Set x-ticks to integers from 1 to 10
plt.grid(True)
plt.show()

# Print the distances for each r value
for r, dist in zip(r_values, distances):
    print(f"Minkowski distance with r={r}: {dist}")


# In[20]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "C:/Users/year3/Downloads/dataset.xlsx"
data = pd.read_excel("C:/Users/year3/Downloads/dataset.xlsx")

# Separate the features and labels
X = data.iloc[:, :-1].values  # All rows, all columns except the last (features)
y = data['LABEL'].values      # All rows, last column (labels)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shape of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[32]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = "C:/Users/year3/Downloads/dataset.xlsx"


try:
    # Load the dataset
    df = pd.read_excel(file_path)
    print("Data loaded successfully.")
    
    # Display the first few rows of the dataframe to understand its structure
    print(df.head())

    # Replace 'target_column' with the actual name of your target column
    # Replace 'feature_columns' with the actual names of your feature columns
    # For example: X = df[['feature1', 'feature2']] and y = df['target_column']
    
    # If column names are numbers or you don't have headers in the file, use index-based access
    # For example: X = df.iloc[:, :-1] and y = df.iloc[:, -1]

    # Replace 'target_column' with the actual column name in your dataset
    target_column = 'target_column'  # Change this to your actual target column name
    feature_columns = df.columns[df.columns != target_column]
    
    X = df[feature_columns]
    y = df[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Optionally, scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the kNN classifier with k=3
    neigh = KNeighborsClassifier(n_neighbors=3)

    # Fit the classifier to the training data
    neigh.fit(X_train, y_train)

    # Evaluate the classifier on the test set
    accuracy = neigh.score(X_test, y_test)
    print(f"Accuracy on the test set: {accuracy:.2f}")

except Exception as e:
    print(f"An error occurred: {e}")


# In[37]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
 
# Load the dataset
file_path = "C:/Users/year3/Downloads/dataset.xlsx"
data = pd.read_excel(file_path)
 
# Identify the class label column and feature columns
class_column = data.columns[-1]
feature_columns = data.columns[:-1]
 
# Get the unique classes in the dataset
unique_classes = data[class_column].unique()
 
# Filter the dataset to include only the first two classes
filtered_data = data[data[class_column].isin(unique_classes[:2])]
 
# Separate features and labels
X = filtered_data[feature_columns].values
y = filtered_data[class_column].values
 
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
# Standardize the features (important for kNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
# Create the kNN classifier with k=3 (you can choose any other value for k)
neigh = KNeighborsClassifier(n_neighbors=3)
 
# Train the classifier
neigh.fit(X_train, y_train)
 
# Test the classifier and get the accuracy
accuracy = neigh.score(X_test, y_test)
 
# Output the accuracy
print(f"Accuracy of kNN on the test set: {accuracy:.2f}")


# In[38]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
 
file_path = "C:/Users/year3/Downloads/dataset.xlsx"
data = pd.read_excel(file_path)
 
# Identify the class label column and feature columns
class_column = data.columns[-1]
feature_columns = data.columns[:-1]
 
# Get the unique classes in the dataset
unique_classes = data[class_column].unique()
 
# Filter the dataset to include only the first two classes
filtered_data = data[data[class_column].isin(unique_classes[:2])]
 
# Separate features and labels
X = filtered_data[feature_columns].values
y = filtered_data[class_column].values
 
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
# Standardize the features (important for kNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
# Create the kNN classifier with k=3 (you can choose any other value for k)
neigh = KNeighborsClassifier(n_neighbors=3)
 
# Train the classifier
neigh.fit(X_train, y_train)
 
# Predict the class for each vector in the test set
y_pred = neigh.predict(X_test)
 
# Output the predictions along with the actual labels
print("Predicted classes for test set:", y_pred)
print("Actual classes for test set:   ", y_test)
 
# To predict a single test vector, e.g., the first one in the test set
test_vect = X_test[0]
predicted_class = neigh.predict([test_vect])
 
print(f"Predicted class for the first test vector: {predicted_class[0]}")
print(f"Actual class for the first test vector: {y_test[0]}")


# In[40]:


import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
 
# Load the dataset

file_path = "C:/Users/year3/Downloads/dataset.xlsx"
data = pd.read_excel(file_path)
 
# Identify the class label column and feature columns

class_column = data.columns[-1]

feature_columns = data.columns[:-1]
 
# Get the unique classes in the dataset

unique_classes = data[class_column].unique()
 
# Filter the dataset to include only the first two classes

filtered_data = data[data[class_column].isin(unique_classes[:2])]
 
# Separate features and labels

X = filtered_data[feature_columns].values

y = filtered_data[class_column].values
 
# Split the data into training and testing sets (80% train, 20% test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
# Standardize the features (important for kNN)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
 
# List to store accuracy results

accuracy_results = []
 
# Iterate over different values of k (from 1 to 11)

for k in range(1, 12):

    # Create the kNN classifier with the current value of k

    neigh = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier

    neigh.fit(X_train, y_train)

    # Test the classifier and get the accuracy

    accuracy = neigh.score(X_test, y_test)

    # Store the accuracy result

    accuracy_results.append(accuracy)
 
    print(f"Accuracy with k={k}: {accuracy:.2f}")
 
# Plot the accuracy results

plt.figure(figsize=(10, 6))

plt.plot(range(1, 12), accuracy_results, marker='o', linestyle='-', color='b')

plt.title('k-NN Accuracy for Different Values of k')

plt.xlabel('k')

plt.ylabel('Accuracy')

plt.xticks(range(1, 12))

plt.grid(True)

plt.show()

 


# In[41]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
 
# Load the dataset
file_path = "C:/Users/year3/Downloads/dataset.xlsx"
data = pd.read_excel(file_path)
 
# Identify the class label column and feature columns
class_column = data.columns[-1]
feature_columns = data.columns[:-1]
 
# Get the unique classes in the dataset
unique_classes = data[class_column].unique()
 
# Filter the dataset to include only the first two classes
filtered_data = data[data[class_column].isin(unique_classes[:2])]
 
# Separate features and labels
X = filtered_data[feature_columns].values
y = filtered_data[class_column].values
 
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
# Standardize the features (important for kNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
# Create the kNN classifier with a chosen k, e.g., k=3
neigh = KNeighborsClassifier(n_neighbors=3)
 
# Train the classifier
neigh.fit(X_train, y_train)
 
# Predict on both training and test data
y_train_pred = neigh.predict(X_train)
y_test_pred = neigh.predict(X_test)
 
# Calculate the confusion matrix for training data
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix (Training Data):\n", conf_matrix_train)
 
# Calculate the confusion matrix for test data
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix (Test Data):\n", conf_matrix_test)
 
# Calculate and print classification report (precision, recall, F1-score) for training data
print("Classification Report (Training Data):")
print(classification_report(y_train, y_train_pred))
 
# Calculate and print classification report (precision, recall, F1-score) for test data
print("Classification Report (Test Data):")
print(classification_report(y_test, y_test_pred))
 
# Evaluate the model's learning outcome based on the metrics
train_accuracy = neigh.score(X_train, y_train)
test_accuracy = neigh.score(X_test, y_test)
 
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
 
if train_accuracy > test_accuracy:
    if test_accuracy < 0.70:  # Assuming a threshold for significant performance drop
        print("The model may be overfitting.")
    else:
        print("The model is well-fitted.")
elif train_accuracy < test_accuracy:
    print("The model may be underfitting.")
else:
    print("The model is well-fitted.")


# In[ ]:





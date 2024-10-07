#A1
import pandas as pd

# Creating the dataset based on the provided table
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data)

# Calculating prior probabilities for each class
total_data_points = len(df)
prior_probabilities = df['buys_computer'].value_counts() / total_data_points

# Displaying the prior probabilities
print("Prior Probabilities for Each Class:")
print(prior_probabilities)




#A2
# Calculating class conditional probabilities for each feature
class_conditional = df.groupby('buys_computer').apply(lambda x: x.apply(lambda y: y.value_counts(normalize=True)))

# Displaying the class conditional probabilities
print("Class Conditional Densities:")
print(class_conditional)




#A3
# Converting categorical data to numerical values for correlation calculation
df_encoded = df.apply(lambda x: pd.factorize(x)[0])

# Calculating the correlation matrix to check independence
correlation_matrix = df_encoded.corr()

# Displaying the correlation matrix
print("Correlation Matrix (Independence Test):")
print(correlation_matrix)




#A4
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Preparing the features and target
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Encoding categorical variables
X_encoded = X.apply(lambda x: pd.factorize(x)[0])
y_encoded = pd.factorize(y)[0]

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)

# Building and training the Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Classifier Accuracy: {accuracy * 100:.2f}%")




#A5
import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Update these paths to the correct locations on your machine
train_dir = r'C:\Users\manik\Downloads\ML DATASET HENRY\Training'  # Update this path
test_dir = r'C:\Users\manik\Downloads\ML DATASET HENRY\Testing'      # Update this path

# Image parameters
image_size = (150, 150)
batch_size = 32

# ImageDataGenerator for loading and preprocessing data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators for training and testing sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Function to extract data and labels from the generator
def extract_data(generator):
    images, labels = [], []
    for _ in range(generator.samples // batch_size):
        img_batch, label_batch = next(generator)  # Use next(generator) to extract batches
        images.extend(img_batch)
        labels.extend(label_batch)
    return np.array(images), np.array(labels)

# Extract training and testing data
X_train, y_train = extract_data(train_generator)
X_test, y_test = extract_data(test_generator)

# Flatten the image data
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=150)  # Keeping 150 principal components
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

# Splitting training data into training and validation sets
X_train_pca, X_val_pca, y_train_pca, y_val_pca = train_test_split(X_train_pca, y_train, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
nb_clf = GaussianNB()
nb_clf.fit(X_train_pca, np.argmax(y_train_pca, axis=1))

# Predict on validation set
y_val_pred = nb_clf.predict(X_val_pca)
val_accuracy = accuracy_score(np.argmax(y_val_pca, axis=1), y_val_pred)
print(f"Validation Accuracy with Naive Bayes after PCA: {val_accuracy:.4f}")

# Predict on test set
y_test_pred = nb_clf.predict(X_test_pca)
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), y_test_pred)
print(f"Test Accuracy with Naive Bayes after PCA: {test_accuracy:.4f}")






import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Defining image parameters
image_size = (150, 150)
batch_size = 32

# Directories
train_dir = r'C:\Users\manik\Downloads\ML DATASET HENRY\Training'
test_dir = r'C:\Users\manik\Downloads\ML DATASET HENRY\Testing'

# ImageDataGenerator for loading training and testing data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Creating generators for training and testing sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Extracting data and labels
def extract_data(generator):
    images, labels = [], []
    for _ in range(generator.samples // batch_size):
        img_batch, label_batch = next(generator)  # Updated to use next(generator)
        images.extend(img_batch)
        labels.extend(label_batch)
    return np.array(images), np.array(labels)

# Extract training and testing data
X_train, y_train = extract_data(train_generator)
X_test, y_test = extract_data(test_generator)

# Flatten the image data for classifiers
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Splitting into training and validation sets
X_train_flat, X_val_flat, y_train_flat, y_val_flat = train_test_split(X_train_flat, y_train, test_size=0.2, random_state=42)

# List of classifiers to compare
classifiers = {
    "Naive Bayes": GaussianNB(),
    "kNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(kernel='linear')
}

# Dictionary to store accuracies
accuracy_dict = {}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train_flat, np.argmax(y_train_flat, axis=1))
    y_pred = clf.predict(X_val_flat)
    accuracy = accuracy_score(np.argmax(y_val_flat, axis=1), y_pred)
    accuracy_dict[name] = accuracy
    print(f"{name} Validation Accuracy: {accuracy:.4f}")

# Find the best classifier based on accuracy
best_classifier = max(accuracy_dict, key=accuracy_dict.get)
print(f"\nBest Classifier: {best_classifier}")

# Use the best classifier to predict on the test set
best_clf = classifiers[best_classifier]
best_clf.fit(X_train_flat, np.argmax(y_train_flat, axis=1))
y_test_pred = best_clf.predict(X_test_flat)
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), y_test_pred)
print(f"\nTest Accuracy with {best_classifier}: {test_accuracy:.4f}")

# Conclusion on whether to use Naive Bayes
if accuracy_dict["Naive Bayes"] == accuracy_dict[best_classifier]:
    print("\nNaive Bayes is the best classifier for your project data.")
elif accuracy_dict["Naive Bayes"] < accuracy_dict[best_classifier]:
    print("\nNaive Bayes is not the best choice for your project data. You should use", best_classifier)
else:
    print("\nConsider using Naive Bayes if model simplicity is a priority.")





import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories
train_dir = r'C:\Users\manik\Downloads\ML DATASET HENRY\Training'
test_dir = r'C:\Users\manik\Downloads\ML DATASET HENRY\Testing'

# Image parameters
image_size = (150, 150)
batch_size = 32

# ImageDataGenerator for loading training and testing data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Creating generators for training and testing sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Function to extract data from the generator
def extract_data(generator):
    images, labels = [], []
    for _ in range(generator.samples // batch_size):
        img_batch, label_batch = next(generator)
        images.extend(img_batch)
        labels.extend(label_batch)
    return np.array(images), np.array(labels)

# Extract training and testing data
X_train, y_train = extract_data(train_generator)
X_test, y_test = extract_data(test_generator)

# Flatten the image data
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Applying PCA to reduce dimensionality
pca = PCA(n_components=150)  # Reducing to 150 principal components
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

# Splitting training data into train and validation sets
X_train_pca, X_val_pca, y_train_pca, y_val_pca = train_test_split(X_train_pca, y_train, test_size=0.2, random_state=42)

# Naive Bayes classifier
nb_clf = GaussianNB()
nb_clf.fit(X_train_pca, np.argmax(y_train_pca, axis=1))

# Predict on validation set
y_val_pred = nb_clf.predict(X_val_pca)
val_accuracy = accuracy_score(np.argmax(y_val_pca, axis=1), y_val_pred)
print(f"Validation Accuracy with Naive Bayes after PCA: {val_accuracy:.4f}")

# Predict on test set
y_test_pred = nb_clf.predict(X_test_pca)
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), y_test_pred)
print(f"Test Accuracy with Naive Bayes after PCA: {test_accuracy:.4f}")

# Conclusion
print("\nAssumption: Naive Bayes assumes that the features are conditionally independent given the class.")
print("If the dataset does not meet this condition, one approach is to use PCA to reduce correlations between features before applying Naive Bayes.")

#A1
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Define paths for training and testing datasets
train_dir = r'C:\Users\manik\Downloads\ML DATASET HENRY\Training'
test_dir = r'C:\Users\manik\Downloads\ML DATASET HENRY\Testing'

# Image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32
input_shape = (img_width, img_height, 3)

# Data augmentation for training set
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, 
                                   horizontal_flip=True)

# Only rescale for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess data
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height),
                                                  batch_size=batch_size, class_mode='categorical', shuffle=False)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4 categories for classification
])

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=test_generator, 
                    steps_per_epoch=train_generator.samples // batch_size, 
                    validation_steps=test_generator.samples // batch_size)

# Predict the classes on the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# True labels
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix')
print(cm)

# Classification report for precision, recall, F1-score
target_names = list(test_generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=target_names)
print('Classification Report')
print(report)

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix, without Normalization")

    print(cm)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

# Plot confusion matrix for visualization
plt.figure(figsize=(8, 8))
plot_confusion_matrix(cm, classes=target_names)
plt.show()


#A3
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# Path to training data (update this path based on your setup)
train_dir = r'C:\Users\manik\Downloads\ML DATASET HENRY\Training'

# Define the classes (subfolders in the training directory)
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Initialize lists to hold feature data (X, Y) and labels
X_data = []
Y_data = []
labels = []

# Function to extract image size as features (X and Y)
def extract_image_features(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize to a fixed size (optional)
    width, height = img.size  # Take width and height as features
    return width, height

# Iterate through classes and images in the training directory
for label, class_name in enumerate(classes):
    class_dir = os.path.join(train_dir, class_name)
    images = os.listdir(class_dir)
    
    # Select 5 random images from each class (to make 20 data points)
    selected_images = random.sample(images, 5)
    
    for image_name in selected_images:
        image_path = os.path.join(class_dir, image_name)
        width, height = extract_image_features(image_path)
        X_data.append(width)
        Y_data.append(height)
        labels.append(label)

# Convert lists to NumPy arrays for easier handling
X_data = np.array(X_data)
Y_data = np.array(Y_data)
labels = np.array(labels)

# Plotting the data
plt.figure(figsize=(8, 6))

# Define colors for each class
colors = ['blue', 'red', 'green', 'purple']

# Scatter plot with class-specific colors
for class_value, color, label_name in zip(range(len(classes)), colors, classes):
    plt.scatter(X_data[labels == class_value], Y_data[labels == class_value], 
                color=color, label=label_name)

plt.title('Scatter Plot of Training Data (Image Width vs Height)')
plt.xlabel('Width')
plt.ylabel('Height')
plt.legend()
plt.grid(True)
plt.show()




#A4
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import os
import random

# Path to training data (update this path based on your setup)
train_dir = r'C:\Users\manik\Downloads\ML DATASET HENRY\Training'

# Define the classes (subfolders in the training directory)
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Initialize lists to hold feature data (X, Y) and labels
X_train = []
Y_train = []
labels = []

# Function to extract image size as features (X and Y)
def extract_image_features(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize to a fixed size (optional)
    width, height = img.size  # Take width and height as features
    return width, height

# Iterate through classes and images in the training directory
for label, class_name in enumerate(classes):
    class_dir = os.path.join(train_dir, class_name)
    images = os.listdir(class_dir)
    
    # Select 5 random images from each class (to create a sample training set)
    selected_images = random.sample(images, 5)
    
    for image_name in selected_images:
        image_path = os.path.join(class_dir, image_name)
        width, height = extract_image_features(image_path)
        X_train.append(width)
        Y_train.append(height)
        labels.append(label)

# Convert lists to NumPy arrays for easier handling
X_train = np.array(X_train)
Y_train = np.array(Y_train)
labels = np.array(labels)

# Prepare the training data
train_data = np.column_stack((X_train, Y_train))

# Step 1: Generate test set data with X, Y values between 0 and 10 with increments of 0.1
x_test = np.arange(0, 10.1, 0.1)
y_test = np.arange(0, 10.1, 0.1)
X_test, Y_test = np.meshgrid(x_test, y_test)
test_data = np.column_stack((X_test.ravel(), Y_test.ravel()))  # Flatten the meshgrid into test data points

# Step 2: Train the kNN classifier (k=3)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(train_data, labels)

# Step 3: Classify the test data points
predictions = knn.predict(test_data)

# Step 4: Plot the test data colored by predicted class
plt.figure(figsize=(10, 8))

# Define colors for each class
colors = ['blue', 'red', 'green', 'purple']

# Plot the test points based on their predicted class
for class_value, color in zip(range(len(classes)), colors):
    plt.scatter(test_data[predictions == class_value][:, 0], 
                test_data[predictions == class_value][:, 1], 
                color=color, label=f'class{class_value}', alpha=0.3, s=10)

plt.title(f'k-NN Classification (k={k}) of Test Data (10,000 points)')
plt.xlabel('X feature')
plt.ylabel('Y feature')
plt.legend()
plt.grid(True)
plt.show()



#A5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import os
import random

# Path to training data (update this path based on your setup)
train_dir = r'C:\Users\manik\Downloads\ML DATASET HENRY\Training'

# Define the classes (subfolders in the training directory)
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Initialize lists to hold feature data (X, Y) and labels
X_train = []
Y_train = []
labels = []

# Function to extract image size as features (X and Y)
def extract_image_features(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize to a fixed size (optional)
    width, height = img.size  # Take width and height as features
    return width, height

# Iterate through classes and images in the training directory
for label, class_name in enumerate(classes):
    class_dir = os.path.join(train_dir, class_name)
    images = os.listdir(class_dir)
    
    # Select 5 random images from each class (to create a sample training set)
    selected_images = random.sample(images, 5)
    
    for image_name in selected_images:
        image_path = os.path.join(class_dir, image_name)
        width, height = extract_image_features(image_path)
        X_train.append(width)
        Y_train.append(height)
        labels.append(label)

# Convert lists to NumPy arrays for easier handling
X_train = np.array(X_train)
Y_train = np.array(Y_train)
labels = np.array(labels)

# Prepare the training data
train_data = np.column_stack((X_train, Y_train))

# Step 1: Generate test set data with X, Y values between 0 and 10 with increments of 0.1
x_test = np.arange(0, 10.1, 0.1)
y_test = np.arange(0, 10.1, 0.1)
X_test, Y_test = np.meshgrid(x_test, y_test)
test_data = np.column_stack((X_test.ravel(), Y_test.ravel()))  # Flatten the meshgrid into test data points

# Define k values to experiment with
k_values = [1, 3, 5, 7, 9]

# Define colors for each class
colors = ['blue', 'red', 'green', 'purple']

# Determine subplot grid dimensions
num_k_values = len(k_values)
num_cols = 3  # Number of columns for subplots
num_rows = int(np.ceil(num_k_values / num_cols))  # Calculate the number of rows needed

# Plot the results for each value of k
plt.figure(figsize=(20, 10))
for i, k in enumerate(k_values):
    # Step 2: Train the kNN classifier with current k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, labels)

    # Step 3: Classify the test data points
    predictions = knn.predict(test_data)

    # Step 4: Plot the test data colored by predicted class
    plt.subplot(num_rows, num_cols, i + 1)
    for class_value, color in zip(range(len(classes)), colors):
        plt.scatter(test_data[predictions == class_value][:, 0], 
                    test_data[predictions == class_value][:, 1], 
                    color=color, label=f'class{class_value}', alpha=0.3, s=10)

    plt.title(f'k-NN Classification (k={k})')
    plt.xlabel('X feature')
    plt.ylabel('Y feature')
    plt.grid(True)

# Adjust layout and show the plots
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()




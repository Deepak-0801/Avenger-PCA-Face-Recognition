import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Directory containing the dataset
data_dir = "dataset/cropped_images/"

# Load images and labels
X = []
y = []
for root, dirs, files in os.walk(data_dir):
    for i, dir in enumerate(dirs):
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith(".png"):
                image_path = os.path.join(root, dir, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (64, 64)) # Resize to 64x64 for consistency
                X.append(image.flatten()) # Flatten the image into a 1D array
                y.append(i) # Assign a label to the image

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Perform PCA
pca = PCA(n_components=50) # Select top 50 principal components
X_pca = pca.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define hyperparameter grid for SVM
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'rbf']}

# Create SVM classifier
svm = SVC()

# Perform grid search to find best hyperparameters
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters: ", best_params)

# Train SVM classifier with best hyperparameters
svm = SVC(**best_params)
svm.fit(X_train, y_train)

# Predict on test set
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy * 100))

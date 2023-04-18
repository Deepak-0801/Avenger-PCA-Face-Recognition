# Avenger-PCA-Facia;-Recognition
 
Image Classification with PCA and SVM
=====================================


This code demonstrates how to perform image classification using Principal Component Analysis (PCA) for dimensionality reduction and Support Vector Machine (SVM) for classification in Python. The code is applied on the Avengers dataset, which consists of images of Avengers characters.

Requirements
------------

The following libraries are required to run the code:

-   os
-   cv2
-   numpy
-   PCA from sklearn.decomposition
-   SVC from sklearn.svm
-   train_test_split and GridSearchCV from sklearn.model_selection
-   accuracy_score from sklearn.metrics

Usage
-----

1.  Download the Avengers dataset or provide your own dataset in the same format.
2.  Install the required libraries, if not already installed.
3.  Load the dataset and preprocess the images using the provided code.
4.  Run the code to apply PCA for dimensionality reduction and SVM for classification.
5.  Experiment with different hyperparameter settings and configurations to improve the model's performance.

Program Details
---------------

The code follows the following steps:

1.  Importing libraries: The necessary libraries are imported, including os, cv2, numpy, PCA and SVC from sklearn.decomposition and sklearn.svm respectively, as well as train_test_split and GridSearchCV from sklearn.model_selection, and accuracy_score from sklearn.metrics.
2.  Loading and preprocessing data: The images and labels are loaded from the dataset directory using os.walk() and cv2 functions. The images are resized to 64x64 pixels and flattened into 1D arrays using image.flatten(), while the labels are assigned to each image based on the subdirectory. The images and labels are converted to numpy arrays.
3.  Performing PCA: PCA is applied to reduce the dimensionality of the image data using PCA from sklearn.decomposition. The top 50 principal components are selected using n_components=50, and the image data is transformed into the reduced feature space using pca.fit_transform().
4.  Splitting the dataset: The PCA transformed data (X_pca) and labels (y) are split into training and testing sets using train_test_split() from sklearn.model_selection. The testing set size is set to 20% of the data with test_size=0.2, and a random state of 42 is used for reproducibility.
5.  Hyperparameter tuning: A hyperparameter grid is defined for SVM with different values of C (penalty parameter of the error term), gamma (kernel coefficient), and kernel (kernel type). The GridSearchCV function from sklearn.model_selection is used to perform a grid search and find the best hyperparameters for the SVM classifier.
6.  Training SVM classifier: An SVM classifier is created using SVC() from sklearn.svm, and the best hyperparameters found in the grid search are passed as arguments using **best_params_. The SVM classifier is trained on the training set using svm.fit().
7.  Making predictions: The trained SVM classifier is used to make predictions on the test data (X_test) using svm.predict(). The predicted labels are stored in y_pred.
8.  Evaluating accuracy: The accuracy of the model is calculated using accuracy_score() from sklearn.metrics, comparing the predicted labels (y_pred) with the true labels (y_test). The accuracy is printed as the final result, indicating the performance of the image classification model.

Results
-------

The accuracy of the image classification model is printed as the final result, indicating the performance of the model on the test data.

Improvements
------------

Users can experiment with different hyperparameter settings and configurations to improve the performance of the model. Additionally, other dimensionality reduction techniques or classifiers can be tried to compare the performance with PCA and SVM.

Conclusion
----------

This code demonstrates how to perform image classification using PCA for dimensionality reduction and SVM for classification in Python, using the Avengers datasetas an example. The code provides a step-by-step implementation of loading and preprocessing the dataset, applying PCA for dimensionality reduction, splitting the data into training and testing sets, hyperparameter tuning for SVM, training the SVM classifier, making predictions, and evaluating the accuracy of the model.

The accuracy of the model can be further improved by experimenting with different hyperparameter settings, exploring other dimensionality reduction techniques, or trying different classifiers. The results of the model can be used for various applications, such as image recognition, object detection, or facial recognition.

Credits
-------

The Avengers dataset used in this code is obtained from [Deepak](https://www.kaggle.com/datasets/rawatjitesh/avengers-face-recognition), and the code is adapted from various sources and tutorials on PCA and SVM in image classification. The code for this program was written by [Deepak](https://github.com/Deepak-0801)

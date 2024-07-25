import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to extract features from images using CNN
def extract_features_with_cnn(image_folder, labels_csv):
    images, labels = []

    labels_df = pd.read_csv(labels_csv)

    for filename in os.listdir(image_folder):
        if filename.endswith(('.JPG', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            label_row = labels_df[labels_df['Image_File'] == filename]

            if not label_row.empty:
                label = float(label_row['Numeric_Label'].values[0])
                img = cv2.imread(image_path)
                img = cv2.resize(img, (128, 128))  # Resize the image
                images.append(img)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Define the CNN model for feature extraction
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Regression output
    ])

    model.compile(optimizer='adam', loss='mse')

    # Normalize image data
    images = images / 255.0

    # Train the model
    model.fit(images, labels, epochs=135, batch_size=16, validation_split=0.2)

    # Extract features from the last dense layer
    feature_extractor = models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = feature_extractor.predict(images)

    return features, labels, feature_extractor

# Provide the paths to your image folder and labels CSV file
image_folder = 'combined_images'
labels_csv = 'new_data.csv'

print("Extracting features using CNN...")

features, labels, feature_extractor = extract_features_with_cnn(image_folder, labels_csv)

# Define the model and parameters
ridge = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}  # Consider a wider range of alpha values

# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define a scoring function for RMSE
scoring = make_scorer(mean_squared_error, squared=False)

# Perform K-Fold cross-validation
cv_results = cross_val_score(ridge, features, labels, cv=kf, scoring=scoring)

# Output cross-validation results
print(f'Cross-validated RMSE: {np.mean(cv_results)}')

# If needed, refit the model on the entire dataset with the best parameters
ridge.fit(features, labels)

# Save the trained Ridge regression model
joblib.dump(ridge, 'models/ridge_regression_model.pkl')

# Save the feature extractor CNN model
feature_extractor.save('models/feature_extractor_model.h5')

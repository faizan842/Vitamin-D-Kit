import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Function to extract features from images using CNN
def extract_features_with_cnn(image_folder, labels_csv):
    images, labels = [], []

    labels_df = pd.read_csv(labels_csv)

    for filename in os.listdir(image_folder):
        if filename.endswith(('.JPG', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            label_row = labels_df[labels_df['Image_File'] == filename]

            if not label_row.empty:
                label = float(label_row['Numeric_Label'].values[0])
                img = cv2.imread(image_path)
                img = cv2.resize(img, (128, 128))  # Resize the image to a smaller size
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
    model.fit(images, labels, epochs=1, batch_size=16, validation_split=0.2)  # Reduce epochs and batch size

    # Extract features from the last convolutional layer
    feature_extractor = models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = feature_extractor.predict(images)

    return features, labels, feature_extractor

# Provide the paths to your image folder and labels CSV file
image_folder = 'combined_images'
labels_csv = 'new_data.csv'

print("Extracting features using CNN...")

features, labels, feature_extractor = extract_features_with_cnn(image_folder, labels_csv)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {'alpha': [0.1]}

# Initialize Ridge regression model
ridge = Ridge()

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_alpha = grid_search.best_params_['alpha']

# Train a Ridge regression model with the best hyperparameters
regressor = Ridge(alpha=best_alpha)
regressor.fit(X_train, y_train)

# Evaluate the model
train_predictions = regressor.predict(X_train)
test_predictions = regressor.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

train_r2_score = r2_score(y_train, train_predictions)
test_r2_score = r2_score(y_test, test_predictions)

print("Best Alpha:", best_alpha)
print("\nTrain RMSE:", train_rmse)
print("Train R-squared Score:", train_r2_score)
print("\nTest RMSE:", test_rmse)
print("Test R-squared Score:", test_r2_score)

# Save the trained Ridge regression model
joblib.dump(regressor, 'models/ridge_regression_model.pkl')

# Save the feature extractor CNN model
feature_extractor.save('models/feature_extractor_model.h5')

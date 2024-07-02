import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Function to load and preprocess images in batches
def load_and_preprocess_images(image_folder, labels_df, batch_size=32):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.JPG', '.jpeg', '.png'))]
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        batch_labels = []
        for filename in batch_files:
            image_path = os.path.join(image_folder, filename)
            label_row = labels_df[labels_df['Image_File'] == filename]
            if not label_row.empty:
                img = cv2.imread(image_path)
                img = cv2.resize(img, (256, 256))
                img = img / 255.0  # Normalize here
                batch_images.append(img)
                batch_labels.append(float(label_row['Numeric_Label'].values[0]))
        yield np.array(batch_images), np.array(batch_labels)

# Function to create the CNN model
def create_cnn_model():
    inputs = Input(shape=(256, 256, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Main function to extract features
def extract_features_with_cnn(image_folder, labels_csv):
    labels_df = pd.read_csv(labels_csv)
    model = create_cnn_model()
    
    # Train the model using fit_generator
    train_generator = load_and_preprocess_images(image_folder, labels_df)
    model.fit(train_generator, steps_per_epoch=len(os.listdir(image_folder))//32, epochs=125, validation_split=0.2)
    
    # Extract features
    feature_extractor = models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    features = []
    labels = []
    for batch_images, batch_labels in load_and_preprocess_images(image_folder, labels_df):
        batch_features = feature_extractor.predict(batch_images)
        features.extend(batch_features)
        labels.extend(batch_labels)
    
    return np.array(features), np.array(labels), feature_extractor

# Main execution
image_folder = 'combined_images'
labels_csv = 'new_data.csv'

features, labels, feature_extractor = extract_features_with_cnn(image_folder, labels_csv)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Use RandomizedSearchCV instead of GridSearchCV
param_distributions = {'alpha': uniform(0.01, 1.0)}
ridge = Ridge()
random_search = RandomizedSearchCV(estimator=ridge, param_distributions=param_distributions, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_alpha = random_search.best_params_['alpha']

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

# Save models
import joblib
joblib.dump(regressor, 'models/ridge_regression_model.pkl')
feature_extractor.save('models/feature_extractor_model.h5')

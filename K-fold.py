import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Function to extract features from images using CNN
def extract_features_with_cnn(image_folder, labels_csv):
    images, labels = [], []  # Initialize two separate lists

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

    images = images / 255.0  # Normalize the images

    model.fit(images, labels, epochs=1, batch_size=16, validation_split=0.2)

    # Extract features from the last convolutional layer
    feature_extractor = models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = feature_extractor.predict(images)

    return features, labels, feature_extractor

# Main script
if __name__ == "__main__":
    # Provide the paths to your image folder and labels CSV file
    image_folder = 'combined_images'
    labels_csv = 'new_data.csv'

    print("Extracting features using CNN...")

    # Call the feature extraction function
    features, labels, feature_extractor = extract_features_with_cnn(image_folder, labels_csv)

    # Define the number of folds
    n_splits = 5

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store scores for each fold
    fold_rmse_scores = []
    fold_r2_scores = []

    # Define the parameter grid for grid search
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

    # Initialize Ridge regression model
    ridge = Ridge()

    # Loop through each fold
    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Perform grid search with cross-validation within the training fold
        grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_alpha = grid_search.best_params_['alpha']

        # Train a Ridge regression model with the best hyperparameters
        regressor = Ridge(alpha=best_alpha)
        regressor.fit(X_train, y_train)

        # Predict on the test set
        test_predictions = regressor.predict(X_test)

        # Evaluate the model
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_r2 = r2_score(y_test, test_predictions)

        fold_rmse_scores.append(test_rmse)
        fold_r2_scores.append(test_r2)

        print(f"Fold {len(fold_rmse_scores)} - Best Alpha: {best_alpha}")
        print(f"Fold {len(fold_rmse_scores)} - RMSE: {test_rmse}")
        print(f"Fold {len(fold_rmse_scores)} - R-squared Score: {test_r2}")

    # Calculate the average scores across all folds
    average_rmse = np.mean(fold_rmse_scores)
    average_r2 = np.mean(fold_r2_scores)

    print("\nAverage RMSE across folds:", average_rmse)
    print("Average R-squared Score across folds:", average_r2)

    # Save the trained Ridge regression model
    joblib.dump(regressor, 'models/ridge_regression_model.pkl')

    # Save the feature extractor CNN model
    feature_extractor.save('models/feature_extractor_model.h5')

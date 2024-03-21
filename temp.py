import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Function to resize, normalize, and apply histogram equalization to images
def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)  # Resize the image
    
    # Convert image to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    img_equalized = cv2.equalizeHist(img)
    
    # Normalize pixel values to be between 0 and 1
    img_normalized = img_equalized / 255.0
    
    return img_normalized.flatten()  # Flatten the image as a feature vector

# Function to load images and labels from a folder and CSV file
def load_images_and_labels(image_folder, labels_csv, target_size):
    images, labels = [], []
    labels_df = pd.read_csv(labels_csv)
    
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            label_row = labels_df[labels_df['Image_File'] == filename]
            if not label_row.empty:
                label = float(label_row['Numeric_Label'].values[0])
                img = preprocess_image(image_path, target_size)
                images.append(img)
                labels.append(label)
    
    return np.array(images), np.array(labels)

# Define CNN model
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))
    return model

# Load images and labels
image_folder = 'cropped kit images'
labels_csv = 'kit-images labels.csv'
target_size = (256, 256)
images, labels = load_images_and_labels(image_folder, labels_csv, target_size)

# Extract features from CNN model
cnn_model = create_cnn_model(input_shape=target_size + (1,))
cnn_features = cnn_model.predict(images.reshape(-1, 256, 256, 1))

# Define the calibration curve equation
def calculate_intensity(concentration):
    intensity = 87.137 * np.exp(-0.017 * concentration)
    return intensity

# Calculate intensity for each label (assuming labels represent concentrations)
intensity_predictions = [calculate_intensity(label) for label in labels]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cnn_features, intensity_predictions, test_size=0.2, random_state=42)

# Train SVR model on combined features
svr_model = make_pipeline(StandardScaler(), SVR(C=100, epsilon=0.01, kernel='rbf'))
svr_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = svr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt

# Input and output directories
input_dir = '/Users/faizanhabib/Desktop/VitaminDkit/Model/cropped_images'
output_dir = '/Users/faizanhabib/Desktop/VitaminDkit/Model/augmented3_images'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Data augmentation
datagen = ImageDataGenerator(
    brightness_range=[0.6, 1.4]  # Adjust brightness
)

# Number of augmented images to generate per input image
num_augmented_images = 4

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        
        # Load the image
        img = load_img(image_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        # Generate augmented images
        augmented_images = datagen.flow(x, batch_size=1)
        
        for i in range(num_augmented_images):
            # Generate one augmented image
            batch = augmented_images.next()
            image_aug = batch[0].astype('uint8')
            
            # Save the augmented image in .JPG format
            save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_augmented3_{i}.JPG")
            plt.imsave(save_path, image_aug, format='jpeg')
            
            print(f"Augmented image {i + 1} saved to {save_path}")

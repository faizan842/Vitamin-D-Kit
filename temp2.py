import cv2
import os

def crop_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Get image dimensions
    height, width, _ = image.shape
    
    # Define cropping margins
    left_margin = int(width * 0.444)
    right_margin = int(width * 0.558)
    top_margin = int(height * 0.49)
    bottom_margin = int(height * 0.541)
    
    # Crop the image
    cropped_image = image[top_margin:bottom_margin, left_margin:right_margin]
    
    return cropped_image

# Define input and output directories
input_directory = 'images'
output_directory = 'cropped_images'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process each image
for filename in os.listdir(input_directory):
    if filename.endswith('.JPG') or filename.endswith('.jpg'):  # Ensure it's an image file
        # Construct full file path
        image_path = os.path.join(input_directory, filename)
        
        # Crop the image
        cropped_image = crop_image(image_path)
        
        # Construct new filename with a consistent format
        new_filename = filename.replace('.JPG', '-cropped.JPG').replace('.jpg', '-cropped.JPG')
        
        # Save the cropped image
        output_image_path = os.path.join(output_directory, new_filename)
        cv2.imwrite(output_image_path, cropped_image)
        
        print(f'Processed: {filename} -> {new_filename}')

print('All images have been processed and cropped.')

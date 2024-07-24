import cv2
import numpy as np

# Load the image
input_image = cv2.imread("output_image.jpeg", cv2.IMREAD_UNCHANGED)

# Check if the image is grayscale or color
if len(input_image.shape) == 2:
    gray_input_image = input_image.copy()
else:
    gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve contour detection
blurred_image = cv2.GaussianBlur(gray_input_image, (5, 5), 0)

# Apply Otsu's thresholding to binarize the image
_, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Check if any contours were found
if contours:
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the object from the image using the bounding box coordinates
    cropped_image = input_image[y:y+h, x:x+w]

    # Save the cropped image
    cv2.imwrite("output_image.jpeg", cropped_image)

    # Optionally display the cropped image
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No object detected in the image.")

import cv2

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

# 1 to 40
for i in range(1, 41):
    image_path = f'new_data/all/#{i}-6000K.JPG'
    cropped_image = crop_image(image_path)
    cv2.imwrite(f'cropped_images/#{i}-6000K.JPG',cropped_image)

# for i in range(1, 41):
#     image_path = f'new_data/#{i}-1800K.JPG'
#     cropped_image = crop_image(image_path)
#     cv2.imwrite(f'cropped_images/{i}-1800K.JPG',cropped_image)

for i in range(1, 41):
    image_path = f'new_data/#{i}-3400K.JPG'
    cropped_image = crop_image(image_path)
    cv2.imwrite(f'cropped_images/{i}-3400K.JPG',cropped_image)

# 61 to 200
for i in range(41,201):
    # image_path = f'new_data/all/{i}a.JPG'
    image_path2 = f'new_data/all/{i}b.JPG'
    image_path3 = f'new_data/all/{i}c.JPG'
    # cropped_image = crop_image(image_path)
    cropped_image2 = crop_image(image_path2)
    cropped_image3 = crop_image(image_path3)
    # cv2.imwrite(f'cropped_images/{i}a.JPG',cropped_image)
    cv2.imwrite(f'cropped_images/{i}b.JPG',cropped_image2)
    cv2.imwrite(f'cropped_images/{i}c.JPG',cropped_image3)

# image_path = '/Users/faizanhabib/Downloads/Cassette Images by Anuja (March 2024) 2/remaining images march 27 2024/IMG_1316.JPG'

# # Crop the image
# cropped_image = crop_image(image_path)

# # # Display the cropped image
# # cv2.imshow('Cropped Image', cropped_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # save the image
# cv2.imwrite('41b.JPG',cropped_image)


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt  # for displaying in Jupyter or similar environments

# def sharpen_image(image):
#     try:
#         kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
#         sharpened_image = cv2.filter2D(image, -1, kernel)
#         return sharpened_image
#     except Exception as e:
#         logging.error(f"Error sharpening image: {e}")
#         raise e

# def crop_image(image_path):
#     try:
#         # Read the image
#         image = cv2.imread(image_path)
        
#         # Get image dimensions
#         height, width, _ = image.shape
        
#         # Define cropping margins
#         left_margin = int(width * 0.444)
#         right_margin = int(width * 0.558)
#         top_margin = int(height * 0.49)
#         bottom_margin = int(height * 0.541)
        
#         # Crop the image
#         cropped_image = image[top_margin:bottom_margin, left_margin:right_margin]
        
#         # Sharpen the cropped image
#         sharpened_image = sharpen_image(cropped_image)
        
#         return sharpened_image
    
#     except Exception as e:
#         logging.error(f"Error cropping and sharpening image: {e}")
#         raise e

# # Example usage
# for i in range(1, 41):
#     image_path = f'new_data/all/#{i}-6000K.JPG'
#     cropped_image = crop_image(image_path)
#     cv2.imwrite(f'cropped_and_sharpened/#{i}-6000K.JPG', cropped_image)

# for i in range(1, 41):
#     image_path = f'new_data/#{i}-3400K.JPG'
#     cropped_image = crop_image(image_path)
#     cv2.imwrite(f'cropped_and_sharpened/{i}-3400K.JPG', cropped_image)

# for i in range(42, 201):
#     image_path2 = f'new_data/all/{i}b.JPG'
#     image_path3 = f'new_data/all/{i}c.JPG'
#     cropped_image2 = crop_image(image_path2)
#     cropped_image3 = crop_image(image_path3)
#     cv2.imwrite(f'cropped_and_sharpened/{i}b.JPG', cropped_image2)
#     cv2.imwrite(f'cropped_and_sharpened/{i}c.JPG', cropped_image3)

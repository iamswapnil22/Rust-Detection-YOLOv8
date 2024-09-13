import cv2
import os
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Function to apply image processing (brightness, contrast, saturation, exposure, zoom, and crop)
def enhance_image(image, brightness=30, contrast=30, saturation=1.5, exposure=1.2, zoom_factor=1.2):
    # Convert to HSV (hue, saturation, value) to manipulate brightness and saturation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Increase brightness
    hsv_image[:, :, 2] = cv2.add(hsv_image[:, :, 2], brightness)
    
    # Increase saturation
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation, 0, 255)
    
    # Convert back to BGR after adjustments
    bright_sat_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    # Adjust contrast and exposure
    enhanced_image = cv2.convertScaleAbs(bright_sat_image, alpha=contrast / 127 + 1, beta=0)
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=exposure)
    
    # Zooming and cropping
    height, width = enhanced_image.shape[:2]
    new_height, new_width = int(height / zoom_factor), int(width / zoom_factor)
    
    top_crop = (height - new_height) // 2
    left_crop = (width - new_width) // 2
    cropped_image = enhanced_image[top_crop:top_crop + new_height, left_crop:left_crop + new_width]
    
    # Resize back to original dimensions
    zoomed_image = cv2.resize(cropped_image, (width, height))
    
    return zoomed_image

# Load the YOLO model
model = YOLO("best100ep.pt")  # Update with your model path

# Path to the directory containing your images
folder_path = r"D:/Swapnil/Swapnil/TEMP Img"

# Create a directory to save the processed images, if needed
output_dir = "runs/detect/predict"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all the files in the folder
for image_file in os.listdir(folder_path):
    # Check if the file is an image (optional)
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        # Full path to the image
        image_path = os.path.join(folder_path, image_file)
        
        print(f"Processing image: {image_file}")
        
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        
        # Apply image enhancement (brightness, contrast, saturation, exposure, and zoom)
        enhanced_image = enhance_image(image)
        
        # Save the processed image (if you want to save the enhanced version)
        enhanced_image_path = os.path.join(output_dir, f"enhanced_{image_file}")
        cv2.imwrite(enhanced_image_path, enhanced_image)
        
        # Use the model to predict on the enhanced image
        results = model.predict(source=enhanced_image_path, save=True, save_txt=True)  # save=True saves the image; save_txt=True saves labels

        # Get the first image from the results (assuming only one image is processed at a time)
        segmented_image = results[0].plot()

        # Display the segmented image using Matplotlib
        # plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')  # Hide axes
        # plt.show()

print(f"Processed images saved in: {output_dir}")
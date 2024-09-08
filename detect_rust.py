# Import required libraries
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
from PIL import Image

# Load the trained YOLOv8 segmentation model
model = YOLO("C:/Users/Lenovo/Downloads/Swapnil/Swapnil/best.pt")  # Update the path if necessary

# Path to the test image
image_path = r"C:\Users\Lenovo\Downloads\Swapnil\Swapnil\tt.jpg"

# Use the model to predict on the image
results = model.predict(source=image_path, save=True)

# Path where the prediction result is saved (update this to match the output path)
prediction_folder = "C:/Users/Lenovo/runs/detect/predict"  # Update the path here

# Find the predicted image in the saved folder
predicted_image_path = os.path.join(prediction_folder, os.path.basename(image_path))

# Check if the image exists and display it
if os.path.exists(predicted_image_path):
    # Open and display the saved image
    img = Image.open(predicted_image_path)
    plt.imshow(img)
    plt.axis('off')  # Hide the axes
    plt.show()
else:
    print(f"Predicted image not found in {predicted_image_path}")

import torch
from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained model
model = YOLO(r"C:\Users\Lenovo\Downloads\Swapnil\Swapnil\best.pt")

# Load the image
img_path = r"C:\Users\Lenovo\Downloads\Swapnil\Swapnil\tt.jpg"
img = cv2.imread(img_path)

# Check if the image was loaded correctly
if img is None:
    raise FileNotFoundError(f"Image file at {img_path} not found or unable to load.")

# Perform inference
results = model(img)

# Inspect results
print("Results:", results)

# Accessing results based on its format
try:
    # Example for the latest format
    boxes = results.pandas().xyxy[0]  # Convert to a pandas DataFrame
    for index, row in boxes.iterrows():
        x1, y1, x2, y2, conf, cls = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{int(cls)} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

except AttributeError:
    print("Error accessing the results. Check the results format and adjust the code accordingly.")

# Display the output
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output
cv2.imwrite("output.jpg", img)

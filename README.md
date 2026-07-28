## 🚀 Model Testing & Quick Start

The custom YOLOv8 weights (`best.pt`) have been fully trained to detect the football, players, and referees. 

To verify the model on a test image, use the following script. 
*(Note: We specifically use `imgsz=1280` to prevent the football from blurring out of existence due to downscaling, and a lower `conf=0.15` to catch the ball during high-speed movement).*

```python
from ultralytics import YOLO
# Note: If running locally instead of Colab, use standard cv2.imshow
from google.colab.patches import cv2_imshow 

# 1. Load the custom trained model
model = YOLO('best.pt')

# 2. Define the path to your test image
image_path = 'test.jpg'

# 3. Run prediction with custom parameters designed for small objects
results = model.predict(image_path, imgsz=1280, conf=0.15, agnostic_nms=True)

# 4. Extract the annotated image array with the bounding boxes
annotated_frame = results[0].plot()

# 5. Display the result
cv2_imshow(annotated_frame)

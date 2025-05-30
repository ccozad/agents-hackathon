from ultralytics import YOLO
import requests
import os.path
import cv2

# Constants for keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

def annotate_pose(image, keypoints, color):
    """
    Add key points to the image for visualization.
    """
    radius = 5
    thickness = 2
    # Enumerate through each tensor in keypoints
    # and draw a circle for each keypoint
    for i in range(len(keypoints)):
        x, y = keypoints[i]
        if i> 4 and x > 0 and y > 0:
            cv2.circle(image, (int(x), int(y)), radius, color, thickness)

colors = [
    (255, 0, 0),   
    (0, 255, 0),   
    (0, 0, 255),   
    (255, 255, 0), 
    (255, 0, 255), 
    (0, 255, 255)  
]

# Load a model
model = YOLO("weights/yolo11n-pose.pt")  # load an official model

image_path = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Kovalev_v_Szilagyi_2013_Fencing_WCH_SMS-IN_t194135.jpg/1024px-Kovalev_v_Szilagyi_2013_Fencing_WCH_SMS-IN_t194135.jpg"
target_image = "image.jpg"
r = requests.get(image_path, stream=True)
with open(target_image, 'wb') as f:
    for chunk in r:
        f.write(chunk)

# Predict with the model
results = model(target_image)  # predict on an image
image = cv2.imread(target_image, cv2.IMREAD_COLOR)

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
    for i, person in enumerate(xy):
        print(f"Processing person {i + 1}")
        annotate_pose(image, person, colors[i % len(colors)])

# Save the annotated image
cv2.imwrite("annotated_pose.jpg", image)

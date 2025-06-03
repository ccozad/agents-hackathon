from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

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

connections = [
    (LEFT_ANKLE, LEFT_KNEE),
    (RIGHT_ANKLE, RIGHT_KNEE),
    (LEFT_KNEE, LEFT_HIP),
    (RIGHT_KNEE, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_SHOULDER),
    (RIGHT_HIP, RIGHT_SHOULDER),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_ELBOW, RIGHT_WRIST),
]

min_area = 15000

colors = [
    (255, 0, 0),   
    (0, 255, 0),   
    (0, 0, 255),   
    (255, 255, 0), 
    (255, 0, 255), 
    (0, 255, 255)  
]

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
    
    # Draw lines between keypoints
    for connection in connections:
        start_index, end_index = connection
        start = keypoints[start_index]
        end = keypoints[end_index]
        if start[0] > 0 and start[1] > 0 and end[0] > 0 and end[1] > 0:
            cv2.line(image, (int(start[0]), int(start[1])),
                     (int(end[0]), int(end[1])), color, 1)

def annotate_bounding_box(image, box, color):
    """
    Draw bounding boxes around detected persons.
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

def area(box):
    """
    Calculate the area of a bounding box.
    """
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

def process_pose_data(image_data):
    """
    Process the image and return a dictionary with pose data.
    """

    pose_data = {
        "metadata": {
            "model": "yolo11n-pose",
            "version": "1.0",
            "description": "Pose estimation data from YOLOv11n model."
        },
        "keypoints": [],
        "bounding_boxes": []
    }

    numpy_image = np.array(image_data)
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    # Load a model
    model = YOLO("weights/yolo11n-pose.pt")  # load an official model 
    # Predict with the model
    results = model(image)  # predict on an image

    # Access the results
    for result in results:
        xy = result.keypoints.xy  # x and y coordinates
        boxes = result.boxes.xyxy  # bounding boxes 
        
        for i, person in enumerate(xy):
            box = boxes[i] if i < len(boxes) else [0, 0, image.shape[1], image.shape[0]]
            area_box = area(box)
            if area_box < min_area:
                continue
            pose_data["keypoints"].append(person.tolist())
            pose_data["bounding_boxes"].append(box.tolist())

    return pose_data


def process_image(image_data):
    """
    Process the image and return the annotated image.
    """

    numpy_image = np.array(image_data)
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    # Load a model
    model = YOLO("weights/yolo11n-pose.pt")  # load an official model 
    # Predict with the model
    results = model(image)  # predict on an image

    # Access the results
    for result in results:
        xy = result.keypoints.xy  # x and y coordinates
        boxes = result.boxes.xyxy  # bounding boxes 
        
        for i, person in enumerate(xy):
            box = boxes[i] if i < len(boxes) else [0, 0, image.shape[1], image.shape[0]]
            area_box = area(box)
            if area_box < min_area:
                continue
            annotate_pose(image, person, colors[i % len(colors)])
            annotate_bounding_box(image, box, colors[i % len(colors)])

    return Image.fromarray(image)


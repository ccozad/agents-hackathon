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

def process_image(image_path, output_path):
    """
    Process the image and return the annotated image.
    """
    # Predict with the model
    results = model(image_path)  # predict on an image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Access the results
    for result in results:
        xy = result.keypoints.xy  # x and y coordinates
        xyn = result.keypoints.xyn  # normalized
        kpts = result.keypoints.data  # x, y, visibility (if available)
        boxes = result.boxes.xyxy  # bounding boxes 
        
        for i, person in enumerate(xy):
            box = boxes[i] if i < len(boxes) else [0, 0, image.shape[1], image.shape[0]]
            area_box = area(box)
            if area_box < min_area:
                continue
            annotate_pose(image, person, colors[i % len(colors)])
            annotate_bounding_box(image, box, colors[i % len(colors)])
    
    cv2.imwrite(output_path, image)

    return image

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

image_paths = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/2013_Nebelhorn_Trophy_Pilar_Maekawa_Moreno_Leonardo_Maekawa_Moreno_IMG_7885.JPG/800px-2013_Nebelhorn_Trophy_Pilar_Maekawa_Moreno_Leonardo_Maekawa_Moreno_IMG_7885.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Sandboarding_in_Dubai.jpg/800px-Sandboarding_in_Dubai.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Kovalev_v_Szilagyi_2013_Fencing_WCH_SMS-IN_t194135.jpg/1024px-Kovalev_v_Szilagyi_2013_Fencing_WCH_SMS-IN_t194135.jpg"
]

for i, image_path in enumerate(image_paths):
    print(f"Downloading {image_path} to image{i}.jpg...")
    target_image = f"image{i}.jpg"
    output_image = f"annotated_image{i}.jpg"
    r = requests.get(image_path, stream=True)
    with open(target_image, 'wb') as f:
        for chunk in r:
            f.write(chunk)

    process_image(target_image, output_image)
    print(f"Annotated image saved as {output_image}")
    

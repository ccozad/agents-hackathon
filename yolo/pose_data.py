from ultralytics import YOLO

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

def print_keypoints(keypoints):
    """
    Print keypoints in a formatted way.
    """
    print(f"Nose: {keypoints[NOSE].tolist()}")
    print(f"Left Eye: {keypoints[LEFT_EYE].tolist()}")
    print(f"Right Eye: {keypoints[RIGHT_EYE].tolist()}")
    print(f"Left Ear: {keypoints[LEFT_EAR].tolist()}")
    print(f"Right Ear: {keypoints[RIGHT_EAR].tolist()}")
    print(f"Left Shoulder: {keypoints[LEFT_SHOULDER].tolist()}")
    print(f"Right Shoulder: {keypoints[RIGHT_SHOULDER].tolist()}")
    print(f"Left Elbow: {keypoints[LEFT_ELBOW].tolist()}")
    print(f"Right Elbow: {keypoints[RIGHT_ELBOW].tolist()}")
    print(f"Left Wrist: {keypoints[LEFT_WRIST].tolist()}")
    print(f"Right Wrist: {keypoints[RIGHT_WRIST].tolist()}")
    print(f"Left Hip: {keypoints[LEFT_HIP].tolist()}")
    print(f"Right Hip: {keypoints[RIGHT_HIP].tolist()}")
    print(f"Left Knee: {keypoints[LEFT_KNEE].tolist()}")
    print(f"Right Knee: {keypoints[RIGHT_KNEE].tolist()}")
    print(f"Left Ankle: {keypoints[LEFT_ANKLE].tolist()}")
    print(f"Right Ankle: {keypoints[RIGHT_ANKLE].tolist()}")


# Load a model
model = YOLO("weights/yolo11n-pose.pt")  # load an official model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
    for i, person in enumerate(xyn):
        print(f"\nPerson {i + 1}:")
        print_keypoints(person)


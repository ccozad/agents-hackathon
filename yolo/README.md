# Introduction

This area of the project is a lab to try out various Ultralytics Yolo v11 workflows.

In the default YOLO11 pose model, there are 17 keypoints, each representing a different part of the human body. Here is the mapping of each index to its respective body joint:

 1. Nose
 2. Left Eye
 3. Right Eye
 4. Left Ear
 5. Right Ear
 6. Left Shoulder
 7. Right Shoulder
 8. Left Elbow
 9. Right Elbow
 10. Left Wrist
 11. Right Wrist
 12. Left Hip
 13. Right Hip
 14. Left Knee
 15. Right Knee
 16. Left Ankle
 17. Right Ankle

 Building a mesh from key points
 - Left leg: left ankle (16) -> left knee (14) -> left hip (12)
 - Right leg: Right ankle (17) -> right knee (15) -> right hip (13)
 - Hips: left hip (12) -> right hip (13)
 - Shoulders: left shoulder (6) -> right shoulder (7)
 - Spine: Hips -> Shoulders (curved line?)
 - Left arm: left wrist (10) -> left elbow (8) -> left shoulder (6)
 - Right arm: right wrist (11) -> right elbow (9) -> right shoulder (7)

# Dependencies

You will need all of the following dependencies to run this example:

 - Install PyTorch with CUDA support
 - Python virtual environment

With Yolo v11 being a state of the art vision model environment there's a long list of dependencies that get installed. PyTorch with GPU support is an enourmous download (GB of data). Vision models with their pre-trained weights are also large downloads. Significant storage space is needed to run the project locally.

## Install PyTorch with CUDA support

Follow the instructions at https://pytorch.org/get-started/locally/

## Python Virtual Environment

 - Move to the gradio hello-world folder
   - `cd <yolo>`
 - Create a virtual environment
   - On Mac: `python3 -m venv .venv`
   - On Windows: `python -m venv .venv`
 - Activate the virtual environment
   - On Mac: `source .venv/bin/activate`
   - On Windows: `.venv\Scripts\activate`
 - Install dependencies
   - On Mac: `pip3 install -r requirements.txt`
   - On Windows: `pip install -r requirements.txt`
 - Call a specific script
   - On Mac: `python3 <script_name>.py`
   - On Windows: `python <script_name>.py`
 - Deactivate virtual environment
   - `deactivate`

# Running the code

## Check PyTorch and GPU support

Run the command `python check_env`

```
python check_env.py
CUDA is available: True
Number of devices: 1
Current device: 0
Active device: <torch.cuda.device object at 0x00000209BB1E70E0>
Device name: NVIDIA GeForce RTX 3060
```

## Pose data example

```
python pose_data.py

Found https://ultralytics.com/images/bus.jpg locally at bus.jpg
image 1/1 D:\Github\agents-hackathon\yolo\bus.jpg: 640x480 4 persons, 111.8ms
Speed: 3.2ms preprocess, 111.8ms inference, 86.8ms postprocess per image at shape (1, 3, 640, 480)

Person 1:
Nose: [0.17573903501033783, 0.4090871810913086]
Left Eye: [0.18268059194087982, 0.39941856265068054]
Right Eye: [0.16114507615566254, 0.40122827887535095]
Left Ear: [0.0, 0.0]
Right Ear: [0.13231153786182404, 0.40798473358154297]
Left Shoulder: [0.19437816739082336, 0.4565742313861847]
Right Shoulder: [0.11637745052576065, 0.4622602164745331]
Left Elbow: [0.21784549951553345, 0.5101668238639832]
Right Elbow: [0.13663692772388458, 0.5255405902862549]
Left Wrist: [0.2150704562664032, 0.4929255545139313]
Right Wrist: [0.20001037418842316, 0.49482154846191406]
Left Hip: [0.18375687301158905, 0.5973376035690308]
Right Hip: [0.12306604534387589, 0.6013010144233704]
Left Knee: [0.22083856165409088, 0.6937926411628723]
Right Knee: [0.11703605949878693, 0.7004795670509338]
Left Ankle: [0.22962231934070587, 0.7868129014968872]
Right Ankle: [0.09130094945430756, 0.7945842742919922]

Person 2:
Nose: [0.0, 0.0]
Left Eye: [0.0, 0.0]
Right Eye: [0.0, 0.0]
Left Ear: [0.0, 0.0]
Right Ear: [0.0, 0.0]
Left Shoulder: [0.999967098236084, 0.45111122727394104]
Right Shoulder: [0.0, 0.0]
Left Elbow: [0.9734852910041809, 0.526656985282898]
Right Elbow: [0.0, 0.0]
Left Wrist: [0.0, 0.0]
Right Wrist: [0.0, 0.0]
Left Hip: [0.9895490407943726, 0.5929509401321411]
Right Hip: [0.988165557384491, 0.5892682075500488]
Left Knee: [0.9486264586448669, 0.6765332818031311]
Right Knee: [0.954727053642273, 0.6739809513092041]
Left Ankle: [0.8973069787025452, 0.7811761498451233]
Right Ankle: [0.0, 0.0]

Person 3:
Nose: [0.36123499274253845, 0.41745370626449585]
Left Eye: [0.3683631718158722, 0.40898141264915466]
Right Eye: [0.34961986541748047, 0.4105246663093567]
Left Ear: [0.3750475347042084, 0.4125397503376007]
Right Ear: [0.3275624215602875, 0.4156584143638611]
Left Shoulder: [0.39277803897857666, 0.45870548486709595]
Right Shoulder: [0.30978628993034363, 0.4622191786766052]
Left Elbow: [0.4059622585773468, 0.5128288269042969]
Right Elbow: [0.30936378240585327, 0.5323130488395691]
Left Wrist: [0.34502899646759033, 0.4953678846359253]
Right Wrist: [0.3117145001888275, 0.5702573657035828]
Left Hip: [0.3746931850910187, 0.5832765102386475]
Right Hip: [0.3201446533203125, 0.5828782916069031]
Left Knee: [0.3717288374900818, 0.6700446009635925]
Right Knee: [0.3223862051963806, 0.6645772457122803]
Left Ankle: [0.3558770716190338, 0.7465158104896545]
Right Ankle: [0.3217242956161499, 0.7455613613128662]

Person 4:
Nose: [0.0, 0.0]
Left Eye: [0.0, 0.0]
Right Eye: [0.0, 0.0]
Left Ear: [0.0, 0.0]
Right Ear: [0.0, 0.0]
Left Shoulder: [0.0, 0.0]
Right Shoulder: [0.0, 0.0]
Left Elbow: [0.0, 0.0]
Right Elbow: [0.0, 0.0]
Left Wrist: [0.0, 0.0]
Right Wrist: [0.0, 0.0]
Left Hip: [0.0, 0.0]
Right Hip: [0.0, 0.0]
Left Knee: [0.0, 0.0]
Right Knee: [0.0, 0.0]
Left Ankle: [0.0, 0.0]
Right Ankle: [0.0, 0.0]
```
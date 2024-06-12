# VisionMouse
VisionMouse: Control Your Cursor with Just Your Eyes 👀 

VisionMouse revolutionizes the way you interact with your computer by using a pre-trained computer vision model to track your eye-gazing and blinking actions. Navigate, click, and perform various functions seamlessly without touching a mouse or trackpad. Perfect for accessibility, hands-free computing, and futuristic tech enthusiasts.

## 1. Concept

Eye_tracking is a method of determining where a person in looking (point of gaze).

### 1.1. Face Detection and Landmark Collection
![face_detect](https://github.com/Havee005/VisionMouse/assets/124234544/3d909d5b-98fa-4aee-9193-5f9ed51ca202)
- Face detection and extracting facial 3D landmark
- Localize the eye area using important landmarks
- Get coordinates for eye patch and crop the required area

### 1.2. Data Collection and Processing
![data_process](https://github.com/Havee005/VisionMouse/assets/124234544/964dcee8-d7be-4fce-acd7-dc5c8308497b)
- 9000 images of faces were taken belomging to 9 classes
- Image size: 50x50x3

### 1.3. Model Intializing, Training and Tuning
![CNN_model](https://github.com/Havee005/VisionMouse/assets/124234544/1a41ea8d-683f-4143-a674-0473eb2cfe89)
- A sequential CNN model used, whivh use ReLU as an activation function.
- Final dense layer uses SoftMax as activation function.

### 1.4. Mouse/ Pointer Control
![cursor](https://github.com/Havee005/VisionMouse/assets/124234544/2466c7fc-4580-48a8-83f2-7d293ec5e817)
- Finally, used cursor library and the CNN prediction to control the cursor.

## 2. Code Explanation

- Used Libraries:
  - CV2 (Open CV)
  - Mediapipe
  - Tensor Flow
  - PyautoGUI (Cursor)
  - sklearn
  - matplotlib
  - numpy 

1. Video_crop:
* Extract the region of interest based on the necessary landmarks.
  
  - -> Input video frame -> detects landmark -> crops the eye area -> Displays the cropped video(process video).

2. Save_img:
* Saves the images in designated folder by capturing the individual frames when spacebar is pressed.
* The images then used to train the CNN model.
* Gets input from the process_video module from Video_crop file.
  
  - -> Input video frame -> Process video -> extract frame -> keyboard event [spacebar]

3. Model_training:
* To train the model copy the path of main folder which contains dataset in code.
* Output -> Weights_of_trained_model.h5 - this file contains the weight of trained model with 75% validation accuracy.

4. Eyestate_predCNN:
* Predicts the class based on the eye movement, uses pre-trained model from CNN training.
* Names of class are pre assigned, so when predicted the respective class's state is published as 'string' for both left_eye & right_eye.
   
   - Gets input from the process_video module from Video_crop file.
		- -> Input video frame -> Process video -> Eye state [model.h5]

6. Eye_blink:
* Tracks the blinking of the eye based of eye aspect ratio (EAR) algorithm.
* The EAR algorithm uses 6 individual landmarks associated to the individual eyes, based on which the blinking ratio is calculated.
* Before that the landmarks are converted to pixel coordinates from normalized coordinates.
  - -> Inpt video frame -> Detect landmark -> Process video -> N2P_Coordinate -> EAR [Euclidean distance] -> Eye state
  - EAR Algorithm: ([|P2-P6|+|P3-P5|]/2[|P1-P4|])

7. Cursor Control:
* Cursor movement according to the eye state.
* Two modules are imported process video from Video crop, eye state from Eyestate_PredCNN.
* Using cursor library.
  - -> Video frame -> eye state -> move cursor [set speed = 50]

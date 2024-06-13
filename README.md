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
  - [CV2 (Open CV)](https://pypi.org/project/opencv-python/)
  - [Mediapipe](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://pypi.org/project/mediapipe/&ved=2ahUKEwjgvrWM2NaGAxUixTgGHVvlCyMQFnoECCMQAQ&usg=AOvVaw2Xtoi8jBb7k9JdNKUSq7Wu)
  - [Tensor Flow](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://pypi.org/project/tensorflow/&ved=2ahUKEwilks6X2NaGAxWy8jgGHZBkCWAQFnoECBUQAQ&usg=AOvVaw2YIamEaRJF6xY2vOSI_Ogb)
  - [PyautoGUI (Cursor)](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://pypi.org/project/PyAutoGUI/&ved=2ahUKEwiavdio2NaGAxWjzzgGHcs9JaQQFnoECCEQAQ&usg=AOvVaw3-mJiYmv6eeguQciD3fJj1)
  - [sklearn](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://pypi.org/project/scikit-learn/&ved=2ahUKEwj5qL-y2NaGAxUlzzgGHYpZB48QFnoECBMQAQ&usg=AOvVaw0o2-kfSWszVppR71GZJXJo)
  - matplotlib
  - numpy

2.1. Video_crop:
* Extract the region of interest based on the necessary landmarks.
  
  - -> Input video frame -> detects landmark -> crops the eye area -> Displays the cropped video(process video).

2.2. Save_img:
* Saves the images in designated folder by capturing the individual frames when spacebar is pressed.
* The images then used to train the CNN model.
* Gets input from the process_video module from Video_crop file.
  
  - -> Input video frame -> Process video -> extract frame -> keyboard event [spacebar]

2.3. Model_training:
* To train the model copy the path of main folder which contains dataset in code.
* Output -> Weights_of_trained_model.h5 - this file contains the weight of trained model with 75% validation accuracy.

2.4. Eyestate_predCNN:
* Predicts the class based on the eye movement, uses pre-trained model from CNN training.
* Names of class are pre assigned, so when predicted the respective class's state is published as 'string' for both left_eye & right_eye.
   
   - Gets input from the process_video module from Video_crop file.
		- -> Input video frame -> Process video -> Eye state [model.h5]

2.5. Eye_blink:
* Tracks the blinking of the eye based of eye aspect ratio (EAR) algorithm.
* The EAR algorithm uses 6 individual landmarks associated to the individual eyes, based on which the blinking ratio is calculated.
* Before that the landmarks are converted to pixel coordinates from normalized coordinates.
  - -> Inpt video frame -> Detect landmark -> Process video -> N2P_Coordinate -> EAR [Euclidean distance] -> Eye state
  - EAR Algorithm: ![image](https://github.com/Havee005/VisionMouse/assets/124234544/146a65ef-cebe-4935-bdba-907776497190)


2.6. Cursor Control:
* Cursor movement according to the eye state.
* Two modules are imported process video from Video crop, eye state from Eyestate_PredCNN.
* Using cursor library.
  - -> Video frame -> eye state -> move cursor [set speed = 50]
 
    
## 3. Result

### 3.1. Face Detection and Eye landmark
	- Used Mediapipe to generate facial landmarks and sticked with necessary keypoints.
![image](https://github.com/Havee005/VisionMouse/assets/124234544/7be2eebc-e996-4606-83bf-e7d5b547f302)

### 3.2. Dataset Collection
	- Appoximately 18000 images were taken, roughly each class contains 1000 images. Used those datas to train the CNN model.
 ![image](https://github.com/Havee005/VisionMouse/assets/124234544/ef8c5760-e04d-4e32-9c34-fee48bcb2f3b)

### 3.3. CNN Accuracy
 	- Achieved 75% accuracy by training the model in Tensor Flow
  ![image](https://github.com/Havee005/VisionMouse/assets/124234544/e0e2413e-3d1e-4008-bbd0-042d8f7513ba)


### 3.4. Cursor Control
https://github.com/Havee005/VisionMouse/assets/124234544/ea36a2a2-cea3-4e99-b541-73cbc4d6f43d

### 3.5. Blink and Clicking
https://github.com/Havee005/VisionMouse/assets/124234544/9b4a897c-af6c-4f98-835e-7cbfbb69e27d

## 4. Dataset and .h5 file
[Dataset](https://drive.google.com/drive/folders/1T9SefQ7yiQSEBzp6S4VYDZY0Lt5DQSXa?usp=drive_link)
[.h5 file](https://drive.google.com/drive/folders/1T9SefQ7yiQSEBzp6S4VYDZY0Lt5DQSXa?usp=drive_link)

## 5. References
1. [Base Paper 1]()
2. [Base Paper 2](https://www.researchgate.net/publication/313449701_Low_cost_eye_based_human_computer_interface_system_Eye_controlled_mouse)
3. [EAR Algorithm](https://ieeexplore.ieee.org/document/9251035)

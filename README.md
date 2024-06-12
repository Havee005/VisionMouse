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

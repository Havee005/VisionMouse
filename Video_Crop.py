import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to detect face landmarks in an image
def detect_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    results = face_mesh.process(image_rgb)    # Extracts 468 Facial Points

    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]  
    else:
        return None

# Function to crop eye patch from the image
def crop_eye_patch(frame, landmarks, eye_landmark_indices, patch_width, patch_height):
    # Convert normalized coordinates to pixel coordinates
    eye_outer_corner = landmarks.landmark[eye_landmark_indices[0]]
    eye_inner_corner = landmarks.landmark[eye_landmark_indices[1]]

    eye_outer_x = int(eye_outer_corner.x * frame.shape[1])
    eye_outer_y = int(eye_outer_corner.y * frame.shape[0])

    eye_inner_x = int(eye_inner_corner.x * frame.shape[1])
    eye_inner_y = int(eye_inner_corner.y * frame.shape[0])

    # Calculate eye patch corner points
    patch_x1 = max(0, min(eye_outer_x, eye_inner_x) - patch_width // 2)
    patch_y1 = max(0, min(eye_outer_y, eye_inner_y) - patch_height // 2)
    patch_x2 = min(frame.shape[1], max(eye_outer_x, eye_inner_x) + patch_width // 2)
    patch_y2 = min(frame.shape[0], max(eye_outer_y, eye_inner_y) + patch_height // 2)

    # Crop eye patch
    eye_patch = frame[patch_y1:patch_y2, patch_x1:patch_x2]
    return eye_patch

# Function to process video frames
def process_video(frame):

    # Initializing with a default value
    left_eye_patch = None  
    right_eye_patch = None

    # Detect landmarks in the frame
    landmarks = detect_landmarks(frame)
    if landmarks:
        # Crop left eye patch
        left_eye_indices = [55, 31]       # [leftEyebrowUpper, leftEyeLower2]
        left_eye_patch = crop_eye_patch(frame, landmarks, left_eye_indices, 25, 25)
        left_eye_patch = cv2.resize(left_eye_patch, (50, 50))
        # Crop right eye patch
        right_eye_indices = [285, 261]    # [rightEyebrowUpper, rightEyeLower2]
        right_eye_patch = crop_eye_patch(frame, landmarks, right_eye_indices, 25, 25)
        right_eye_patch = cv2.resize(right_eye_patch, (50, 50))

    # uncomment while testing:
        # Save The Image (Test)
        # cv2.imwrite("left_eye_patch.jpg", left_eye_patch)
        # cv2.imwrite("right_eye_patch.jpg", right_eye_patch)

        # Display frames
    #     cv2.imshow('Original Video', frame)
    #     cv2.imshow('Left Eye Patch', left_eye_patch)
    #     cv2.imshow('Right Eye Patch', right_eye_patch)
    # else:
    #     cv2.imshow('Original Video', frame)
    
    return frame, left_eye_patch, right_eye_patch

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        

    while True:
        ret, frame = cap.read()  
        if not ret:
            print('No input from ret')
            continue

        frame = cv2.flip(frame, 1)
        # Output Frames
        process_video(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    

"""
    leftEyeUpper0: [246, 161, 160, 159, 158, 157, 173],
    leftEyeLower0: [33, 7, 163, 144, 145, 153, 154, 155, 133],
    leftEyeUpper1: [247, 30, 29, 27, 28, 56, 190],
    leftEyeLower1: [130, 25, 110, 24, 23, 22, 26, 112, 243],
    leftEyeUpper2: [113, 225, 224, 223, 222, 221, 189],
    leftEyeLower2: [226, 31, 228, 229, 230, 231, 232, 233, 244],
    leftEyeLower3: [143, 111, 117, 118, 119, 120, 121, 128, 245],  ## <-- USED

    leftEyebrowUpper: [156, 70, 63, 105, 66, 107, 55, 193],   ## <-- USED
    leftEyebrowLower: [35, 124, 46, 53, 52, 65],

    rightEyeUpper0: [466, 388, 387, 386, 385, 384, 398],
    rightEyeLower0: [263, 249, 390, 373, 374, 380, 381, 382, 362],
    rightEyeUpper1: [467, 260, 259, 257, 258, 286, 414],
    rightEyeLower1: [359, 255, 339, 254, 253, 252, 256, 341, 463],
    rightEyeUpper2: [342, 445, 444, 443, 442, 441, 413],
    rightEyeLower2: [446, 261, 448, 449, 450, 451, 452, 453, 464],
    rightEyeLower3: [372, 340, 346, 347, 348, 349, 350, 357, 465],  ## USED

    rightEyebrowUpper: [383, 300, 293, 334, 296, 336, 285, 417],   ## USED
    rightEyebrowLower: [265, 353, 276, 283, 282, 295]

    RIGHT_IRIS = [474,475, 476, 477]
    LEFT_IRIS = [469, 470, 471, 472]
"""
import cv2
import mediapipe as mp
import math

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
    
# Function to find Euclidean Distance
def Euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def N2P_coordinates(image, landmarks, eye_landmark_indices):
    keypoints = []

    # Convert normalized coordinates to pixel coordinates for each landmark index
    for idx in eye_landmark_indices:
        landmark_point = landmarks.landmark[idx]
        x = int(landmark_point.x * image.shape[1])
        y = int(landmark_point.y * image.shape[0])
        keypoints.append((x, y))
    # print(keypoints)
    return keypoints


def EAR(keypoints):
    P1, P2, P3, P4, P5, P6 = keypoints

    vert1 = Euclidean_distance(P2, P6)
    vert2 = Euclidean_distance(P3, P5)
    horiz = Euclidean_distance(P1, P4)

    calc_ratio = ((vert1 + vert2) / (2*horiz)) * 10
    return calc_ratio

def eye_state(left_ratio, right_ratio):
    # R_thres = 1.9
    # L_thres = 2.9
    # print (left_ratio, right_ratio)
    # Thresholding value fixed based on trial and error 
    if  left_ratio < 2 and right_ratio < 2.0:
        state = 'Closed'
    elif right_ratio < 2 and left_ratio > 2:
        state = 'Right Eye Is Closed'
    elif left_ratio < 2.9 and right_ratio < 2.4:
        state = 'Left Eye Is Closed'
    else:
        state = 'Open'

    return state
    
def process_video(frame):

    # Initializing with a default value
    left_keypoints= None  
    right_keypoints = None

    # Detect landmarks in the frame
    landmarks = detect_landmarks(frame)
    if landmarks:
        # Left_keypoints  
        left_eye_indices = [246, 160, 158, 173, 153, 163]  # [P1, P2, P3, P4, P5, P6]
        left_keypoints = N2P_coordinates(frame, landmarks, left_eye_indices)
        left_ratio = EAR(left_keypoints)

         # Right_keypoints
        right_eye_indices = [446, 387, 385, 398, 380, 390] # [P1, P2, P3, P4, P5, P6]
        right_keypoints = N2P_coordinates(frame, landmarks, right_eye_indices)
        right_ratio = EAR(right_keypoints)

        state = eye_state(left_ratio, right_ratio)
        cv2.putText(frame, state, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Left: {left_ratio:.2f}, Right: {right_ratio:.2f}", (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 191), 2)

    cv2.imshow('video',frame)  

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
    # out.release()
    cv2.destroyAllWindows()
    face_mesh.close()
import cv2
import os
import mediapipe as mp
from Video_Crop import process_video


# Function to extract and save frames
def extract_frame(frame, folder_name, counter):
    # Create the output folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    output_path = os.path.join(folder_name, f"frame_{counter}.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Frame {counter} saved.")
    return counter + 1

# Function to handle keyboard events
def keyboard_event(event, output_folders,counters, frame, roi_left, roi_right):
    if event == 32:  # Spacebar key code
        # Save original frame
        counters['Face_frame'] = extract_frame(frame, output_folders['Face_frame'], counters['Face_frame'])  
        # Save roi_left
        counters['left'] = extract_frame(roi_left, output_folders['left'], counters['left'])  
        # # Save roi_right
        counters['right'] = extract_frame(roi_right, output_folders['right'], counters['right'])  

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

    # Folders to save extracted frames
    main_folders = ["Face_frame", "left", "right"]

    # Initialize counters for each main folder
    counters = {folder:549 for folder in main_folders}

    # Create subfolders 'A - I' within each main folder
    # Change the name of the folder when needed
    output_folders = {folder: os.path.join(folder, "I") for folder in main_folders} # <-- change name here 
    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)

    # Display Before Begin
    print("Press 'q' to exit (or) Press Space Bar to save images")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No input from camera")
            break

        frame = cv2.flip(frame, 1)

        frame, left_eye_patch, right_eye_patch = process_video(frame)
        cv2.imshow('Original Video', frame)
        cv2.imshow('Left Eye Patch', left_eye_patch)
        cv2.imshow('Right Eye Patch', right_eye_patch)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        else:
            keyboard_event(key, output_folders, counters, frame, left_eye_patch, right_eye_patch)
    
    cap.release()
    cv2.destroyAllWindows()
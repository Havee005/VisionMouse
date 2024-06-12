import cv2
import pyautogui
from Video_Crop import process_video
from Eyestate_PredCNN import eye_state

# Set the speed of cursor movement
CURSOR_SPEED = 50

# Function to move the cursor based on the eye state
def move_cursor(left_eye_class, right_eye_class):
    screen_width, screen_height = pyautogui.size()
    current_x, current_y = pyautogui.position()
    new_x, new_y = current_x, current_y
    
    # Move the cursor based on the eye state
    if left_eye_class == 'Up':
        new_y = max(0, current_y - CURSOR_SPEED)
    elif left_eye_class == 'Down':
        new_y = min(screen_height, current_y + CURSOR_SPEED)
    if right_eye_class == 'Left':
        new_x = max(0, current_x - CURSOR_SPEED)
    elif right_eye_class == 'Right':
        new_x = min(screen_width, current_x + CURSOR_SPEED)
        
    # Move the cursor
    pyautogui.moveTo(new_x, new_y, duration=0.2)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        
    frame_counter = 0
    while True:
        ret, frame = cap.read()  
        if not ret:
            print('No input from ret')
            continue

        frame_counter += 1
        if frame_counter % 5 != 0:  # Skip processing some frames
            continue

        frame = cv2.flip(frame, 1)
        frame, left_eye_patch, right_eye_patch = process_video(frame)

        # Normalize the pixel values to the range [0, 1]
        left_eye_pixels = left_eye_patch / 255.0  
        right_eye_pixels = right_eye_patch / 255.0 
        
        # Get eye state
        _, left_eye_class, right_eye_class = eye_state(frame, left_eye_pixels, right_eye_pixels)

        # Move cursor based on eye state
        move_cursor(left_eye_class, right_eye_class)
        
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

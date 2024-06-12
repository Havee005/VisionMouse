import cv2
import tensorflow as tf
import numpy as np
from Video_Crop import process_video

# Predict eye state using the model
def eye_state(frame, left_image, right_image):

    # Load the trained TensorFlow model
    model = tf.keras.models.load_model('Weights_of_trained_model.h5')

    # Class names
    class_names = ['Down', 'Up', 'Up_left', 'Up_right', 'Low_left', 'Low_right', 'Left', 'Right', 'Middle']

    left_eye_state = model.predict(np.expand_dims(left_image, axis=0))[0]
    right_eye_state = model.predict(np.expand_dims(right_image, axis=0))[0]
    
    left_eye_class = class_names[np.argmax(left_eye_state)]
    right_eye_class = class_names[np.argmax(right_eye_state)]

    cv2.putText(frame, f"Left Eye: {left_eye_class}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Right Eye: {right_eye_class}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('frame',frame)
    return frame, left_eye_class, right_eye_class


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
        frame, left_eye_patch, right_eye_patch = process_video(frame)

        # Normalize the pixel values to the range [0, 1]
        left_eye_pixels = left_eye_patch / 255.0  
        right_eye_pixels = right_eye_patch / 255.0 
        
        eye_state(frame, left_eye_pixels, right_eye_pixels)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

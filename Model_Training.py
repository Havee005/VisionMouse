import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, Accuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Hyperparameters
img_width, img_height = 50, 50
num_classes = 9
batch_size = 40
epochs = 25
learning_rate = 0.001



# Load and preprocess the training data
def load_and_preprocess_data(directory):
    X = []
    y = []

    # Mapping subfolder names to labels for the right and left classes
    right_eye = {
        'RIGHT_DOWN': 0,
        'RIGHT_UP': 1,
        'RIGHT_UPLEFT': 2,
        'RIGHT_UPRIGHT': 3,
        'RIGHT_LOWLEFT': 4,
        'RIGHT_LOWRIGHT': 5,
        'RIGHT_LEFT': 6,
        'RIGHT_RIGHT': 7,
        'RIGHT_MID': 8
         }


    left_eye = {
        'LEFT_DOWN': 0,
        'LEFT_UP': 1,
        'LEFT_UPLEFT': 2,
        'LEFT_UPRIGHT': 3,
        'LEFT_LOWLEFT': 4,
        'LEFT_LOWRIGHT': 5,
        'LEFT_LEFT': 6,
        'LEFT_RIGHT': 7,
        'LEFT_MID': 8
        }

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                # Read the image
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                print("Image Path:", image_path)
                print("Image Shape:", image.shape)
               
                if image is not None:
                    # Resize the image
                    image = cv2.resize(image, (img_width, img_height))
                    # Normalize pixel values
                    image = image / 255.0
                    # Append to X
                    X.append(image)
                    # Get the class label from the folder name
                    class_name = os.path.basename(os.path.dirname(image_path))
                    print("Class Name:", class_name)
                    if "right_eye" in image_path:
                        label = right_eye[class_name]
                    elif "left_eye" in image_path:
                        label = left_eye[class_name]
                    else:
                        # If class name does not match, skip the image
                        continue
                    print("Label:", label)
                    y.append(label)
                    # X.append(image)
                    # y.append(label)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("No data loaded. Check data loading and preprocessing.")

    return np.array(X), np.array(y)

def create_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(img_width, img_height, 3 )))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  
    model.compile(loss=CategoricalCrossentropy(),
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[Precision(), Recall(), Accuracy()])
    return model




# Define paths to the dataset
train_dir = "F:/Assignments/Computer Vision/CV_Projectt/CV_Project_VisionMouse/main"

X, y = load_and_preprocess_data(train_dir)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for CNN input
X_train = X_train.reshape(-1, img_width, img_height, 3)
X_val = X_val.reshape(-1, img_width, img_height, 3)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)

# Data Generator for training with augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
rotation_range=20,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
# horizontal_flip=True,
# rescale=1./255
)

# Data Generator for validation without augmentation
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=50)

validation_generator = validation_datagen.flow(X_val, y_val, batch_size=50)


model = create_model()





# Train the model
history = model.fit(X_train, y_train,
                    
                    batch_size=batch_size,

                    epochs=epochs,
                    validation_data=(X_val, y_val))




# Save the trained model
model.save('Weights_of_trained_model.h5')

# Print training history
print("Training Accuracy:", history.history['accuracy'][-1]) 
print("Validation Accuracy:", history.history['val_accuracy'][-1])

# Get predictions on validation data
predictions = model.predict(X_val)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_val, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, range(num_classes), rotation=45)
plt.yticks(tick_marks, range(num_classes))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

accuracy = accuracy_score(true_labels, predicted_labels)



# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

# Plot accuracy graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot recall graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Recall over epochs')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.show()

# Plot precision graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Precision over epochs')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.show()









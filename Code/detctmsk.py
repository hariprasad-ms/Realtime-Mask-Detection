# import libraries
import tensorflow as tf
import cv2
import numpy as np

# Define CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=100, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=100, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(35, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the labels
labels_dict = {0: 'No mask', 1: 'Mask'}

# Define the colors for the labels
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

# Set image resize
img_size = 4

# Turn on the camera
camera = cv2.VideoCapture(0)

# Load the face classifier
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the loop
while True:
    # Read the camera feed
    (rval, im) = camera.read()
    # Flip the image
    im = cv2.flip(im, 1, 1)
    # Resize the image
    imgs = cv2.resize(im, (im.shape[1] // img_size, im.shape[0] // img_size))
    # Detect faces
    face_rec = classifier.detectMultiScale(imgs)
    # Iterate over the faces
    for i in face_rec:
        # Get the coordinates and size of the face
        (x, y, l, w) = [v * img_size for v in i]
        # Get the face image
        face_img = im[y:y+w, x:x+l]
        # Resize the face image
        resized = cv2.resize(face_img, (150, 150))
        # Normalize the face image
        normalized = resized / 255.0
        # Reshape the face image
        reshaped = np.reshape(normalized, (1, 150, 150, 3))
        # Predict the label of the face
        result = model.predict(reshaped)
        # Get the predicted label
        label = np.argmax(result, axis=1)[0]
        # Draw a rectangle around the face
        cv2.rectangle(im, (x, y), (x+l, y+w), color_dict[label], 2)
        # Draw a rectangle for the label
        cv2.rectangle(im, (x, y-40), (x+l, y), color_dict[label], -1)
        # Put the label on the image
        cv2.putText(im, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Show the image
    cv2.imshow('LIVE', im)
    # Wait for a key press
    key = cv2.waitKey(10)
    # If the Esc key is pressed, break the loop
    if key == 27:
        break

# Release the camera
camera.release()

# Destroy all windows
cv2.destroyAllWindows()

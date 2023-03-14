<a href="#"><img width="100%" height="auto" src="https://github.com/hariprasad-ms/Realtime-Mask-Detection/blob/main/Assets/GitWall.png" height="175px"/></a>

<p align="center">
    <img alt="Build" src="https://img.shields.io/badge/build-passed-success">
    <img alt="Contributors" src="https://img.shields.io/badge/contributors-1-blue">
    <img alt="Status" src="https://img.shields.io/badge/status-working-success">
    <img alt="Status" src="https://img.shields.io/badge/progress-integrating-important">
</p>

# Real-Time Mask Detection Using Python And Deep Learning (cv2)

`This repository contains the my research project reguarding mask detection as an aspect for the new feature development of Destiny, as well as for Personal Knowledge.`

---

This project is an example of Deep Learning, specifically Convolutional Neural Networks (CNNs). Here we uses a pre-trained CNN model to perform face mask detection on live video feed from a camera.

CNNs are a type of neural network that are commonly used for image classification tasks, and they have been proven to be highly effective in many computer vision applications. The CNN model used in this project has been trained on a large dataset of images to recognize faces with and without masks, and it is used to make predictions on new images in real-time.

---

## Working

The project contains a Python script that uses Keras and OpenCV libraries to perform face mask detection on live camera feed using a pre-trained convolutional neural network (CNN). The script can be divided into several sections as follows:

<br />

> **Install The Prerequisite**

**Pip Command to install opencv-python**

```bash
pip install opencv-python

```

**Pip Command to install tensorflow**

```bash
pip install tensorflow

```

<br />

> **Importing Required Libraries**

```python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

import cv2
import numpy as np

```

This section imports the required libraries for the script, including TensorFlow and Keras libraries for building and training the CNN model, OpenCV for image processing, and NumPy for numerical computations.

<br />

> **Building and Compiling CNN Model**

```python

cnn = Sequential([Conv2D(filters=100, kernel_size=(3,3), activation='relu'),
                   MaxPooling2D(pool_size=(2,2)),
                   Conv2D(filters=100, kernel_size=(3,3), activation='relu'),
                   MaxPooling2D(pool_size=(2,2)),
                   Flatten(),
                   Dropout(0.5),
                   Dense(50),
                   Dense(35),
                   Dense(2)])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


```
This section defines the CNN model architecture and compiles the model. The model consists of two convolutional layers, two max-pooling layers, a flatten layer, three dense layers, and a dropout layer. The model is compiled with the Adam optimizer, binary cross-entropy loss function, and accuracy as the evaluation metric.

<br />

> **Defining Labels and Colors for Visualization**

```python

labels_dict={0:'No mask', 1:'Mask'}
color_dict={0:(0,0,255), 1:(0,255,0)}


```

This section defines a dictionary for the labels of the output classes and a dictionary for the colors of the bounding boxes and text overlays.

<br />

> **Initializing Camera and Face Detection Classifier**


```python

imgsize = 4 #set image resize
camera = cv2.VideoCapture(0) # Turn on camera
# Identify frontal face
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```
This section sets the image resize factor, initializes the camera, and loads the face detection classifier for detecting the faces in the live video feed.

<br />

> **Processing Video Feed and Performing Face Mask Detection**

```python

while True:
    (rval, im) = camera.read()
    im=cv2.flip(im,1,1) #mirrow the image
    imgs = cv2.resize(im, (im.shape[1] // imgsize, im.shape[0] // imgsize))
    face_rec = classifier.detectMultiScale(imgs) 
    for i in face_rec: # Overlay rectangle on face
        (x, y, l, w) = [v * imgsize for v in i] 
        face_img = im[y:y+w, x:x+l]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=cnn.predict(reshaped)
        label=np.argmax(result,axis=1)[0]
        cv2.rectangle(im,(x,y),(x+l,y+w),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+l,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.imshow('LIVE',im)
    key = cv2.waitKey(10)
    # stop loop by ESC
    if key == 27: # The Esc key
        break
```

This section of the code processes the live video feed and performs face mask detection using the pre-trained CNN model. It consists of a while loop that runs continuously until the user terminates the program by pressing the Esc key.

Within the while loop, the code reads a frame from the camera, resizes the image to a smaller size, and applies the face detection classifier to detect the faces in the image. For each detected face, the code extracts the face region, resizes it to 150x150, normalizes the pixel values, and feeds it to the CNN model for prediction.

After obtaining the prediction from the model, the code overlays a rectangle around the detected face and displays the predicted class label on top of the rectangle. The label can be "Mask" or "No mask" depending on whether the model predicts that the face is wearing a mask or not. The code also uses different colors for the rectangle and the text based on the predicted label.

The output of the face mask detection process is displayed in real-time using the `cv2.imshow()` function, which shows the processed image with the detected faces and predicted labels overlaid. Finally, the code waits for 10 milliseconds for a key press event, and if the key pressed is the Esc key, the program is terminated using the `cv2.destroyAllWindows()` function.

---

## Test Results

> **`Testing on self target for mask on`** 
    <hr></hr>
    <a><img width="100%" height="auto" src="https://github.com/hariprasad-ms/Realtime-Mask-Detection/blob/main/Result/YesMask.png" height="175px"/></a>
    <details><summary>Read more...</summary></br>
    <p>It is hereby shown that the model was able to sucessfully predict that there is mask on my face.<hr></hr></p></details>
    
<br />

> **`Testing on self target for mask off`** 
    <hr></hr>
    <a><img width="100%" height="auto" src="https://github.com/hariprasad-ms/Realtime-Mask-Detection/blob/main/Result/NoMask.png" height="175px"/></a>
    <details><summary>Read more...</summary></br>
    <p>It is hereby shown that the model was able to sucessfully predict that there is no mask on my face<hr></hr></p></details>

---

## Supported Environments

|                         |                                         |
|-------------------------|-----------------------------------------|
| **Operating systems**   | Linux & Windows                         |
| **Python versions**     | Python 3.7.6 (64-bit)                   |
| **Distros**             | Ubuntu, Windows 8, 8.1 Pro, 10 (All Distros)         |
| **Package managers**    | APT, pip                                |
| **Languages**           | English                                 |
| **System requirements** | 4GB of free RAM, Intel i3 - Any Higher  |
|                         |                                         |

---

### Do Checkout My other recent [`Research Projects`]() as well as [`Project Destiny`](https://github.com/Our-Destiny)

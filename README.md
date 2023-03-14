<a href="#"><img width="100%" height="auto" src="https://github.com/hariprasad-ms/Realtime-Gender-Detection/blob/main/Assets/GitWall.png" height="175px"/></a>

<p align="center">
    <img alt="Build" src="https://img.shields.io/badge/build-passed-success">
    <img alt="Contributors" src="https://img.shields.io/badge/contributors-1-blue">
    <img alt="Status" src="https://img.shields.io/badge/status-working-success">
    <img alt="Status" src="https://img.shields.io/badge/progress-integrating-important">
</p>

# Real-Time Gender Detection Using Python And Deep Learning (cv2)

`This repository contains the my research project reguarding gender detection as an aspect for the development of Destiny, as well as for Personal Knowledge.`

---

This project is made using a deep learning-based model for face and gender detection provided by the cvlib 
library.
Specifically, the cvlib library is built on top of OpenCV and uses pre-trained deep learning models for object
detection tasks like face detection and gender detection.

---

## Working

The project contains a Python script that performs real-time gender detection using the computer vision libraries OpenCV and cvlib.

First, it imports the necessary libraries, `including cvlib, cv2, and numpy.`

> **Install The Prerequisite**

**Pip Command to install opencv-python**

```bash
pip install opencv-python

```

It then initializes the webcam object by calling `cv2.VideoCapture(0)`, which sets the default camera device as the video source. It also sets the padding value to 20, which is used later to add a padding around the detected face region to ensure that the entire face is captured in the frame.

Next, the script enters a while loop that reads each frame from the video feed using the `webcam.read()` method. For each frame, the `cv.detect_face()` method of the cvlib library is called to detect any faces in the frame. If a face is detected, the code draws a green rectangle around the face using the `cv2.rectangle()` method of OpenCV.

Then, the script crops the detected face region using the `np.copy()` method and the coordinates of the rectangle that were detected. It then passes this cropped face region to the `cv.detect_gender()` method of the cvlib library to predict the gender of the person in the face. The method returns two lists: label and confidence, which contain the predicted gender labels and the confidence scores, respectively.

The code then selects the gender label with the highest confidence score using the `np.argmax()` function. It constructs a string label using the predicted gender label and the confidence score, and adds this label to the frame using the `cv2.putText()` method.

Finally, the code displays the processed frame using the `cv2.imshow()` method and waits for the user to press the "s" key to stop the program. If the "s" key is pressed, the program releases the camera using the `webcam.release()` method and destroys all windows created by OpenCV using the `cv2.destroyAllWindows()` method.

Overall, this code uses computer vision libraries to perform real-time gender detection from a video feed, and demonstrates how these libraries can be used to analyze and process video data in real-time.

---

## Test Results

> **`Testing on self target for mask on`** 
    <hr></hr>
    <a><img width="100%" height="auto" src="https://github.com/hariprasad-ms/Realtime-Mask-Detection/blob/main/Result/YesMask.png" height="175px"/></a>
    <details><summary>Read more...</summary></br>
    <p>It is hereby shown that the model was able to sucessfully predict if iam a male or female. The percentage value visible is the confidence of prediction.<hr></hr></p></details>
    
<br />

> **`Testing on self target for mask off`** 
    <hr></hr>
    <a><img width="100%" height="auto" src="https://github.com/hariprasad-ms/Realtime-Mask-Detection/blob/main/Result/NoMask.png" height="175px"/></a>
    <details><summary>Read more...</summary></br>
    <p>It is hereby shown that the model was able to sucessfully predict if iam a male or female. The percentage value visible is the confidence of prediction.<hr></hr></p></details>

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

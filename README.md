## Introduction
It does face recognition and parameterization to recognize all faces in videos. Uses the DBScan clustering technique to cluster the faces recognized from the video. The clustered faces and Euclidean distance matrix is used to identify the known faces in a new video/camera feed.

## Getting Started
Installation: 

– CUDA and cudNN for NVIDIA GPU support.

– Dlib library for CNN-based face detection.

– mmod_human_face_detector.dat for face parameterization.

## Key Features
It detects faces in each frame of the video. For each face detected, it assigns 128 face parameters and makes a pickle file.

Then, it groups similar faces into clusters based on the closeness of the Euclidean distance of the face parameters. It uses the DBScan face clustering technique to do the same. It makes another pickle file with the clustered faces.

It detects faces in an unknown video, matches those faces with the clustered faces pickle file, and displays the text "SAME FACE" for known faces.

## Advanced Functionalities
This does not require supervised training for known faces. Therefore, a huge dataset with multiple images of the same person is not required to be created manually. It uses clustering to identify unique faces in unseen videos. These unique faces are then detectable in other unseen videos. 

## Tech Stack
Programming Language: Python

Libraries used:

1.sklearn (scikit learn)

2.imutils

3.numpy

4.argparse

5.pickle

6.OpenCV (cv2)

7.sys 

8.dlib

9.time

10.face_recognition

## How To Run
There are 3 main codes. 

– The cnn_face_encoder.py file is used to detect faces in each frame of the video and make a pickle file out of it. 

– The clusterFaces.py file is used to group similar faces into classes and make another pickle file of clustered faces. 

– The comperator_actual.py file is used to compare the faces in the new video with the faces the machine was trained with (detection). The pickle files and mmod_human_face_detector.dat must be linked at the necessary places.

Terminal commands are as follows:

Training: python cnn_face_encoder.py -i ./vdos/train.mp4 -m ./mmod_human_face_detector.dat -e ./encodings/aachar.pickle -d ./faceDump/dumping_face

Clustering: python clusterFaces.py -e encodings/aachar.pickle -d 1 -o encodings/aachar-clustered.pickle

Detecting (testing): python comperator_actual.py -i vdos/test.mp4 -m mmod_human_face_detector.dat -e encodings/aachar-clustered.pickle

## Applications
– It can be used for proxy detection in educational institutions and workplaces as face detection for marking attendance would be a strict system.

– It can used to detect missing children from CCTV camera footage of different streets which they might be suspected to have taken.

– It can be used for detecting hostages from the videos released by kidnappers which will help the police in tracking them down.

## Further Improvements
The face clustering code groups faces into different classes. There can be cases where the side profile and the front profile of the same person is put into different classes and we manually have to group them as the same person. I would like to improve on this so that the machine itself can group the front and side profiles of a person and put them into a single class.

## Resources
Inspiration was taken from the blog on face detection by pyimagesearch. An example code from dlib library was enhanced for this project.

## Demo Video

[![Watch the video](https://img.youtube.com/vi/e6uIOivYKd4/0.jpg)](https://www.youtube.com/watch?v=e6uIOivYKd4&ab_channel=GautamKappagal)  

🔹 **Timestamp:** [6:30 - 9:30](https://www.youtube.com/watch?v=e6uIOivYKd4&t=390s)  


# Driver_drowsiness_system_CNN
This is a system which can detect the drowsiness of the driver using CNN - Python, OpenCV

The aim of this is system to reduce the number of accidents on the road by detecting the drowsiness of the driver and warning them using an alarm. 

Here, we used Python, OpenCV, Keras(tensorflow) to build a system that can detect features from the face of the drivers and alert them if ever they fall asleep while while driving. The system dectects the eyes and prompts if it is closed or open. If the eyes are closed for 3 seconds it will play the alarm to get the driver's attention, to stop cause its drowsy.We have build a CNN network which is trained on a dataset which can detect closed and open eyes. Then OpenCV is used to get the live fed from the camera and run that frame through the CNN model to process it and classify wheather it opened or closed eyes.

## Setup
To set the model up:
Pre-install all the required libraries 1) OpenCV__
                                       2) Keras__
                                       3) Numpy__
                                       4) Pandas__
                                       5) OS__
                                       
Run the Jupyter Notebook and add the file name in detect_drowsiness.py line 20.

## The Dataset
The dataset which was used is a subnet of a dataset from(https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)
it has 4 folder which are 1) Closed_eyes - having 726 pictures
                          2) Open_eyes - having 726 pictures
                          3) Yawn - having 725 pictures
                          4) no_yawn - having 723 pictures

## The Convolution Neural Network

## Accuracy 
We did 50 epochs, to get a good accuracy from the model i.e. 98% for training accuracy and 96% for validation accuracy.

## The Output 

  [




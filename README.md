# Driver_drowsiness_system_CNN
This is a system which can detect the drowsiness of the driver using CNN - Python, OpenCV

The aim of this is system to reduce the number of accidents on the road by detecting the drowsiness of the driver and warning them using an alarm. 

Here, we used Python, OpenCV, Keras(tensorflow) to build a system that can detect features from the face of the drivers and alert them if ever they fall asleep while while driving. The system dectects the eyes and prompts if it is closed or open. If the eyes are closed for 3 seconds it will play the alarm to get the driver's attention, to stop cause its drowsy.We have build a CNN network which is trained on a dataset which can detect closed and open eyes. Then OpenCV is used to get the live fed from the camera and run that frame through the CNN model to process it and classify wheather it opened or closed eyes.

## Setup
To set the model up:<br />
Pre-install all the required libraries <br />1) OpenCV<br />
                                       2) Keras<br />
                                       3) Numpy<br />
                                       4) Pandas<br />
                                       5) OS<br />
Download the Dataset from the link given below and edit the address in the notebook accordingly.<br />
Run the Jupyter Notebook and add the model name in detect_drowsiness.py file in line 20.<br />

## The Dataset
The dataset which was used is a subnet of a dataset from(https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)<br />
it has 4 folder which are <br />1) Closed_eyes - having 726 pictures<br />
                          2) Open_eyes - having 726 pictures<br />
                          3) Yawn - having 725 pictures<br />
                          4) no_yawn - having 723 pictures<br />

## The Convolution Neural Network
![CNN](https://user-images.githubusercontent.com/16632408/159187014-4bc4b70e-98d6-4313-873f-997ded2eff27.png)

## Accuracy 
We did 50 epochs, to get a good accuracy from the model i.e. 98% for training accuracy and 96% for validation accuracy.
![Graph](https://user-images.githubusercontent.com/16632408/159187004-92a72662-ddfe-471d-8bd6-65a3593a70a1.png)

## The Output 
1. Open Eyes<br />
![Open_eyes](https://user-images.githubusercontent.com/16632408/159187179-b557ab8e-fb8c-4408-850b-417893014f8c.png)
2. Close Eyes<br />
Here we detect wheater the eyes are closed and count the number of frames for which the eyes were closed (which is 10 frame) greater then that the Alarm will ring and the WARNING sign is displayed.
![Closed_eyes](https://user-images.githubusercontent.com/16632408/159187305-68cbdee3-8325-4216-85e3-7dbb66a429fb.png)



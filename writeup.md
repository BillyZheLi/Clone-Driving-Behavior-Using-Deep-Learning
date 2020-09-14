# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The steps of this project are the following:
* Used the simulator to collect data of good driving behavior
* Built a convolution neural network in Keras that predicts steering angles from images
* Trained and validated the model with a training and validation set
* Tested that the model successfully drives around track one without leaving the road
* Summarized the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model being used is publised by the autonomous vehicle team in Nvidia. The model begings with preprocessing, including a normalization layer which will normalized each pixel value bewteen -0.5 and 0.5 (code line 55), and a cropping layers which crops the top 60 rows of the image as well as the bottom 20 rows of the image (code line 56). Following the preprocessing are five convolution layers, each of the convolutional layer includes RELU layers to introduce nonlinearity (code line 57-61). The output is then flattened (code line 62) and connected to four fully connected layers (code line 63-66). Since this is a regression network, the model ouputs one value which is the predicted steering angle.

#### 2. Attempts to reduce overfitting in the model

In order to fully take advantage of the powerful network, I tried no dropout layer first. But I tried to train the model with different number of epochs (code line 74) and see after how many epochs the validation loss wil start to increase. After a few trials, I chose four epochs and this will effectively reduce the overfitting of the model.

Another strategy I have used is to collect more traing data. I recorded my driving for 2 laps in total, one is clockwise and the other is counter-clockwise.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line 69).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving smoothly around curves. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to gradually increasing the complexity of the model while watching its behaviour by runing the autonomous mode.

The first step was to use only one fully connected layer, just to verify that I can train, save, and deploy a model. After the verification step, I implemented the model published by the autonomous vehicle team in Nvidia.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that after four epochs, my model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model such that only four epochs are used instead of ten epochs. I also collected more training data by recording one more lap of my driving one the track but in the opposite direction.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track around the curves. To improve the driving behavior in these cases, I collected more data where I deliberately drove to the side of the road, started recording, driving back to the center of the road, and stopped recording.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 54-66) consisted of a convolution neural network with the following layers and layer sizes.

Layer | Description
--- | --- 
Input | 160 * 320 * 3 RGB file
Normalization | Normalize to [-0.5, 0.5]
Cropping | Crop top 60 and bottom 20 rows
Convolution 5 * 5 | 2 * 2 strides, 24 filters
Relu | 
Convolution 5 * 5 | 2 * 2 strides, 36 filters
Relu | 
Convolution 5 * 5 | 2 * 2 strides, 48 filters
Relu | 
Convolution 3 * 3 | 1 * 1 strides, 64 filters
Relu | 
Convolution 3 * 3 | 1 * 1 strides, 64 filters
Relu | 
Flatten | 
Fully connected | outputs 100 nodes
Fully connected | outputs 50 nodes
Fully connected | outputs 10 nodes
Fully connected | outputs 1 nodes (the prediction)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text](https://github.com/BillyZheLi/Clone-Driving-Behavior-Using-Deep-Learning/blob/master/selected%20images/center_2020_09_14_03_13_55_676.jpg "Model Visualization")

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it drives off the center. These images show what a recovery (from right side) looks like

![alt text](https://github.com/BillyZheLi/Clone-Driving-Behavior-Using-Deep-Learning/blob/master/selected%20images/center_2020_09_14_03_14_04_792.jpg "Recovery Image")
![alt text](https://github.com/BillyZheLi/Clone-Driving-Behavior-Using-Deep-Learning/blob/master/selected%20images/center_2020_09_14_03_14_05_005.jpg "Recovery Image")
![alt text](https://github.com/BillyZheLi/Clone-Driving-Behavior-Using-Deep-Learning/blob/master/selected%20images/center_2020_09_14_03_14_06_006.jpg "Recovery Image")
![alt text](https://github.com/BillyZheLi/Clone-Driving-Behavior-Using-Deep-Learning/blob/master/selected%20images/center_2020_09_14_03_14_07_050.jpg "Recovery Image")
![alt text](https://github.com/BillyZheLi/Clone-Driving-Behavior-Using-Deep-Learning/blob/master/selected%20images/center_2020_09_14_03_14_08_033.jpg "Recovery Image")
![alt text](https://github.com/BillyZheLi/Clone-Driving-Behavior-Using-Deep-Learning/blob/master/selected%20images/center_2020_09_14_03_14_08_595.jpg "Recovery Image")

To augment the data sat, I also flipped images and angles thinking that this would generalize the training set. For example, here is an image that has then been flipped:

![alt text](https://github.com/BillyZheLi/Clone-Driving-Behavior-Using-Deep-Learning/blob/master/selected%20images/center_2020_09_14_03_14_11_344.jpg "Unflipped Image")
![alt text](https://github.com/BillyZheLi/Clone-Driving-Behavior-Using-Deep-Learning/blob/master/selected%20images/center_2020_09_14_03_14_11_344_flipped.jpg "Flipped Image")

After the collection process, I had 16072 number of images and measurements. I then preprocessed this data by normalizing each pixel to [-0.5,0.5] and then cropping the top 60 and botton 20 rows of each image.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as the validation loss will decrease for the frist 4 epochs and then increase after four or five epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary. The figure below showed the traing loss and validation loss after each of the epoch.

![alt text](https://github.com/BillyZheLi/Clone-Driving-Behavior-Using-Deep-Learning/blob/master/selected%20images/Figure_1.png "Training and validation loss")

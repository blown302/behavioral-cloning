# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with:

- Normalization layer.
- Cropping layer to reduce noise outside the region of interest.
- 5 convolutional layers with relu activation each layer.
  - 24 filters with 5x5 kernel and 2x2 stride.
  - 32 filters with 3x3 kernel and 2x2 stride.
  - 48 filters with 3x3 kernel and 2x2 stride.
  - 64 filters with 3x3 kernel and 1x1 stride.
  - 64 filters with 3x3 kernel and 1x1 stride.
- Flatten layer. 
- 3 fully connected layers with relu activation.
  - Dense layer with 100 units.
  - Dense layer with 50 units.
  - Dense layer with 10 units.
- Dense output layer with 1 output for steering angle.

RELU activation are used throughout the graph to introduce non-linearity.

#### 2. Attempts to reduce overfitting in the model

Started with using dropout but but did not seem necessary when using a larger stride in the initial convolutional layers.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, driving on the edges of the road and slower driving to during maneuvers.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start off with the suggested arch and tweak depending
performance.

My first step was to start with the suggested Nvidia architecture. 

I saw that the car would have a hard time with the first long sweeping corner.
Iteratively added different features to the data when appropriate. First tried the first 3 convolutions with `1x1`
strides but that made my model extremely large and hard to manage/low performance. Tried to apply max pooling to reduce
this size after the third layer. To reduce overfitting I added a few dropout layers. After switching from max pooling 
to `2x2` strides it seemed like dropouts were not a necessary.

After each iteration I added data and more types of data, trained model, evaluated validation with training via visualization,
and ran a test with the simulator.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

- Normalization layer with dims (160, 320, 3).
- Cropping layer to reduce noise outside the region of interest with dims (60, 320, 3).
- 5 convolutional layers with relu activation each layer.
  - 24 filters with 5x5 kernel, 2x2 stride and dims (28, 158, 24).
  - 32 filters with 3x3 kernel, 2x2 stride and dims (13, 78, 32).
  - 48 filters with 3x3 kernel, 2x2 stride and dims (6, 38, 48).
  - 64 filters with 3x3 kernel, 1x1 stride and dims (4, 36, 64).
  - 64 filters with 3x3 kernel, 1x1 stride and dims (2, 34, 64).
- Flatten layer with size 4352. 
- 3 fully connected layers with relu activation.
  - Dense layer with size 100.
  - Dense layer with size 50.
  - Dense layer with size 10.
- Dense output layer with 1 output for steering angle.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

For slight error correction I recorded right lane driving and left lane driving.

To generalize the model I repeated the above but going clockwise. Then added the right and left cameras with an offset
to augment the steering angle. To generalize further flipped the images to keep the model from being biased to left
turns. 

After testing the model continued to have an issue with the dirt patch after bridge as if it considered it a path to
drive rather than the edge of the track. To mitigate this I tried different color spaces like HSV and RGB. RGB seemed
be most effective.

Finally, not make the tight turns so I tried tweaking the right and left camera offset but made it swerve excessively.
I set the offset low to prevent swerving and then added an augmentation to increase the steering angle for all examples. 

After this iterative process I ended with 35497 data points. Some of these points were from the other track.
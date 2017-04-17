# **Behavioral Cloning** 
Ricardo Solano
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture.png "Model Visualization"
[image2]: ./examples/center_2017_04_14_18_30_03_056.jpg "Center lane driving"
[image3]: ./examples/center_2017_04_14_15_57_08_553.jpg "Recovery Image"
[image4]: ./examples/center_2017_04_14_15_57_08_787.jpg "Recovery Image"
[image5]: ./examples/center_2017_04_14_15_57_08_949.jpg "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (clone.py lines 76-90) 

The image pixel data is normalized and mean centered in the model using a Keras lambda layer (code line 78). Input images are cropped 70 pixels from the top and 20 from the bottom in order to eliminate unnecessary details that could potentially distract the model. This is achieved via a Keras Cropping2D layer.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets in order to check that the model was not overfitting (code line 105). In order to reduce overfitting, Dropout layers were introduced after each fully connected layer (code lines 85, 87, 89). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 96).

#### 4. Appropriate training data

I generated approximately 7 different data sets in order to train the model. I used a combination of center lane driving, recovering from the left and right sides of the road back to the center, and driving in both directions.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create data sets and experimenting with various models. I started with the simplest network, a flattened image connected to a single output, to check everything was working. Obviously no amount of data would make this model drive correctly so I moved on to a LeNet architecture (http://yann.lecun.com/exdb/lenet/). LeNet did much better and I was able to get the car to drive itself past the bridge. However it would not get past the second curve and veer off into the dirt, regardless of the amount of data that I fed the architecture with. I created data sets by driving on the center lane and focusing getting through curves as smoothly as possible. I also recorded center recoveries from both sides of the road and different points of the circuit, emphasizing problem areas such as the bridge and the second curve. I also recorded clockwise and counter-clockwise driving to help the mode better generalize the track.

To help with training performance I wrote a generator function. The generator create and yields data on the fly with a batch size of 32. This greatly improved my ability to test out different data sets and models, even without a GPU.

Lastly I implemented Nvidia's network architecture (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) which was designed for real self-driving vehicles. Given its complexity I anticipated training would be time consuming but that was not the case to my surprise. I first trained the model for 3 epochs and the car already drove better than all my previous attempts with LeNet, even using my smallest data set. After training for 5 epochs, the car was able to drive around the entire track without leaving the road.

#### 2. Final Model Architecture

My final model was based on Nvidia's architecture, a convolutional neural network which consists of the following layers and layer sizes: 

| Layer | Description	| 
|:-----:|:-----------:| 
| Input | 320x160x3 RGB image |
| Cropping | outputs 320x90x3 |
| Normalization | normalize to 0-1 and mean center to 0-0.5 |
| Convolution 5x5	|	outputs 66x200x24	|
| Convolution 5x5	|	outputs 31x98x24	|
| Convolution 5x5	|	outputs 14x47x36	|
| Convolution 5x5	|	outputs 5x22x48	|
| Convolution 5x5	|	outputs 1x18x64	|
| Flatten | outputs 1164 |
|	Fully connected |	outputs 100 |
|Dropout|
|	Fully connected |	outputs 50 |
|Dropout|
|	Fully connected |	outputs 10 |
|Dropout|
|	Fully connected |	outputs 1 |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn not to drive over the lane lanes and move back to the center instead.

The images below show what a recovery looks like starting from the left side toward the center:

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data set, I flipped the images and angles in order to help the model generalize the track. Image flipping was done on the fly using the cv2.flip() function (lines, 47, 50 and 53) and not persisted to filesystem.

My final training data set had 9300 data points after the collection process. I then preprocessed this data normalizing and mean centering pixel data and cropping each image. Data set is randomly shuffled with 20% of it put into the validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by a validation loss decrease up to that point and by the model's ability to correctly drive around the track.

# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test_images/hist.jpg "Visualization"
[image2]: ./test_images/original.jpg "Original Images"
[image3]: ./test_images/normalized.jpg "Normalized Images"
[image4]: ./test_images/image1.jpg "Traffic Sign 1"
[image5]: ./test_images/image2.jpg "Traffic Sign 2"
[image6]: ./test_images/image3.jpg "Traffic Sign 3"
[image7]: ./test_images/image4.jpg "Traffic Sign 4"
[image8]: ./test_images/image5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This markdown file is the Writeup report on German Traffic Sign Recognition project.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram chart showing how the training data is distributed among different classes. As we can see, some of the classes have more training images than other. Which can affect the training process.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

For this project, I decided only to normalize the pixel values between **[0, 1]** by dividing each image pixel with **255**. Because grayscaling the images removes the  color channels, all the information stored in colours will get lost. So, I decided not to convert the images to grayscale.

Here is an example of an original image and an augmented image:

**Original Images:**

![alt text][image2]

**Normalized Images:**

![alt text][image3] 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 Coloured image   					    | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16	|
| Dropout               |                                               |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64   |
| Dropout               |                                               |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				    |
| Fully connected		| With 1600x128 weigth and 128 bias parameters  |
| Dropout               |                                               |
| RELU                  |                                               |
| Fully connected		| With 128x64 weigth and 64 bias parameters     |
| Dropout               |                                               |
| RELU                  |                                               |
| Fully connected		| With 64x43 weigth and 43 bias parameters      |
| Softmax				|           									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the **AdamOptimizer**. It has better performance than gradient descent optimizer. It uses moving averages of the parameters (momentum). The **Batch Size I used is 128**, to train the model more quickly. Since the availibility of processing power. I have used **20 epochs** to iterate over the whole dataset to train the model. The **learning rate used was 0.001**.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.57 %
* validation set accuracy of 95.12 %
* test set accuracy of 94.92 %

If a well known architecture was chosen:
* What architecture was chosen?

The Architecthre I choose is LeNet architecture. Which is simple but very powerful Convolutional Neural Network Model for medium sized image dataset. It worked really well with German Traffic Sign dataset. I have choosen a learning rate of 0.001 and a total of 20 epochs.

* Why did you believe it would be relevant to the traffic sign application?

The LeNet architecture model is a simple but very powerful Model for medium sized image dataset like German Traffic Sign dataset.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The LeNet model provides an accuracy of 95.12 % on validation set and 94.92 % on Test set, which are very close to each other. The model provides almost same level of accuracy for each set which indicates that the model is working well.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

1. **Stop Sign:**

![alt text][image4] 

2. **Speed Limit(30 km/h):**

![alt text][image5] 

3. **Keep Right:** 

![alt text][image6] 

4. **Road Work:**

![alt text][image7]

5. **Speed Limit(60 km/h):**

![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Speed Limit (30 km/h) | Speed Limit (30 km/h)                         |
| Keep Right			| Keep Right									|
| Road Work	      		| Road Work					 				    |
| Speed Limit (60 km/h)	| Speed Limit (60 km/h)      					|


The model was able to correctly guess all 5 of the traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.92 %.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 22th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a 30km/h Speed limit sign (probability of 0.193), and the image does contain the same sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .193         			| Speed limit (30km/h)   						| 
| .119    				| General caution 								|
| .100					| Roundabout mandatory							|
| .067	      			| Speed limit (20km/h)					 		|
| .065				    | Speed limit (50km/h)      					|


For the second image, the model is relatively sure that this is a 60km/h Speed limit sign (probability of 6.57e-01), and the image does contain the same sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 6.57e-01         		| Speed limit (60km/h)   						| 
| 3.39e-01     			| Speed limit (80km/h) 							|
| 2.40e-03				| No passing for vehicles over 3.5 metric tons	|
| 4.55e-04      		| No passing					 				|
| 3.04e-04				| Vehicles over 3.5 metric tons prohibited     	|


For the third image, the model is relatively sure that this is a Road work sign (probability of 9.98e-01), and the image does contain the same sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.98e-01      		| Road work  									| 
| 1.68e-03     			| Bicycles crossing 							|
| 1.67e-06				| Beware of ice/snow							|
| 1.00e-06	    		| Bumpy Road					 				|
| 6.82e-07			    | Priority road     							|


For the fourth image, the model is relatively sure that this is a Keep right sign (probability of 9.99e-01), and the image does contain the same sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99e-01              | Keep right   									| 
| 4.14e-05    			| Turn left ahead 								|
| 1.81e-06				| Roundabout mandatory							|
| 1.31e-07	      		| Priority road					 				|
| 1.30e-07				| Keep left      							    |


For the fifth image, the model is relatively sure that this is a Stop sign (probability of 1.00e+00), and the image does contain the same sign. The top five soft max probabilities were  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00e+00       		| Stop         									| 
| 1.54e-08    			| No entry 										|
| 5.44e-09				| No vehicles									|
| 4.34e-10	      		| Bicycles crossing					 			|
| 2.03e-10				| Speed limit (80km/h)      					|

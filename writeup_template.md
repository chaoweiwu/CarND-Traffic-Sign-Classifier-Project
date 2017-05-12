#**Traffic Sign Recognition** 

##Writeup

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[all_labels]:./writeup_images/all_labels_displayed.png "All traffic signs displayed"
[distplot]: ./writeup_images/distplot.png "Distribution of training images"
[web_images]: ./writeup_images/web_images.png "All the traffic signs from the web"
[prediction_confidence]: ./writeup_images/prediction_confidence.png "Confidence of predictions"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

The size of training set is 34799
The size of test set is 12630
The shape of a traffic sign image is (32, 32, 3), where 32 by 32 represents the pixel locations and 3 represents the number of color channels.
The number of unique labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

I visualized the dataset by displaying one example of each sign type titled with the label number and name. I used matplotlib's `imshow` function.
I also visualized distribution of labels in the dataset using `sns.distplot`

![Distribution of training images][distplot]
![Examples of all the labels][all_labels]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I preprocessed the images by normalizing the pixel data by taking the mean and standard deviation of each pixel channel. 
Then I subtracted each image channel datapoint by its corresponding channel mean and divided by it's corresponding channel deviation.
This process occurs in `main.preprocess_multichannel`

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I did not do any additional work the data as it was already provided in three sets `train.p`, `valid.p`, `test.p`. 
I did not do any additional work to augment the data.
There were 34799, 34799, and 12630 images in the train, validation and test data sets respectively.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in architecture.py. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5X5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6  				    |
| Flatten               | outputs 400x1                                 |
| Fully connected		| Outputs hidden layer size 120        	        |
| Dropout               | keep_prob = .5 during training                |
| RELU                  |                                               |
| Fully connected		| Outputs hidden layer size 84        			|
| Dropout               | keep_prob = .5 during_training                |
| RELU                  |                                               |
| Fully connected		| Outputs classes size 43        				|
| Dropout               | keep_prob = .5 during_training                |
| Softmax				| Returns probability of each class        		|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in model.py

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.978
* validation set accuracy of 0.934
* test set accuracy of 0.933

I used the LeNet Architecture with added Dropout in the dense layers. I chose to test it because it worked well 
in the minst lab, and after seeing it gave promising results, I decided to keep it and tune it. 
An important aspect to this architecture is dropout, as it will prevent the model from overfitting previously seen images.
I can tell that the model is working overall because it is giving over a 93% accuracy on test data that it has not seen before.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
![Web traffic sign examples][web_images]

It's worth noting there was noise on most of the photos, because they were watermarked stock photos.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Sixty km/h            | Sixty km/h   									| 
| Ahead Only     		| Ahead Only 								    |
| Children Crossing	    | Stop											|
| Roundabout Mandatory	| Roundabout Mandatory					 		|
| No Entry			    | No Entry      							    |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 
This compares unfavorably to the accuracy on the test set of .9333. This could result from watermarks on the images.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

![Confidence of predictions for traffic signs from the web][prediction_confidence]:

PATTERN RECOGNITION AND MACHINE LEARNING
KNN-CLASSIFIER
GROUP 16

This project contains one folder and four files:

DATA FOLDER
Contains contains data used for the project in .mat files. Both raw and preprocessed data can be found. 
preprocessed_numberdata.mat:
	- pre-processed data for the model 
	- labels of the data
	- boolean array according to which the data can be split into training and validation sets

example_knn_model.mat:
	- knn_model struct which contains the training data, its labels and k value used in training

FILES
digit_classify.m: main classifier function, which calls other function needed for classification

knn_classify.m: the knn algorithm

preprocess_data: data preprocessing function
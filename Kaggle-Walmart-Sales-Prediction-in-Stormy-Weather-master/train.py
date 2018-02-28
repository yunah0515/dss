''' Predictioin and Evaluation '''

import time
import numpy as np 
import pandas as pd
import data_preprocess
import prediction_model
start = time.time()
print "Training Start..."


item_list = data_preprocess.get_item_list()

for item_nbr in item_list:
	''' Get the training dataset '''
	X_train, y_train = data_preprocess.split_item_data(item_nbr, 'train')

	''' Scale the training dataset '''
	X_train = prediction_model.scale_data(X_train, item_nbr)

	''' Train the model '''
	# Linear Regression
	#prediction_model.linear_regression_train(X_train, y_train, item_nbr)
	#Lasso
	#prediction_model.lasso_train(X_train, y_train, item_nbr)
	#SVM
	svm_kernel = 'rbf'
	prediction_model.svm_train(X_train, y_train, item_nbr, svm_kernel)


end = time.time()
print "Training End..."
print "Running time: %f" %(end-start)
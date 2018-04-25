''' Prediction model to predict the sales based on weather data'''

import math
import numpy as np 
from sklearn import svm, linear_model, cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

''' Compute Root Mean Squared Logarithmic Error (RMSLE) '''
def rmsle_metrics(y_true, y_pred):
	error = 0.0
	for i in range(len(y_true)):
		#print y_true[i], y_pred[i]
		temp = math.pow(math.log(y_true[i]+1) - math.log(y_pred[i]+1), 2)
		error += temp
	rmsle_error = math.sqrt(error/len(y_true))
	return rmsle_error

''' Center the data and scale to unit variance '''
def scale_data(X_train, item_nbr):
	clf = StandardScaler()
	scalar = clf.fit(X_train)
	X_train = scalar.transform(X_train)
	filename = "scalar/scale_%d.pkl" %(int(item_nbr))
	joblib.dump(scalar, filename)
	return X_train

#####################################################################################

#####################################################################################
def linear_regression_train(X_train, y_train, item_nbr):
	clf = linear_model.LinearRegression()
	clf.fit(X_train, y_train)
	filename = "model/LR_model_%d.pkl" %(int(item_nbr))
	joblib.dump(clf, filename)

def linear_regression_test(X_test, item_nbr):
	scalar_name = "scalar/scale_%d.pkl" %(int(item_nbr))
	scalar = joblib.load(scalar_name)
	X_test = scalar.transform(X_test)
	
	model_name = "model/LR_model_%d.pkl" %(int(item_nbr))
	clf = joblib.load(model_name)
	y_pred = clf.predict(X_test)
	return y_pred
#####################################################################################
#####################################################################################
def lasso_train(X_train, y_train, item_nbr):
	clf = linear_model.Lasso()
	clf.fit(X_train, y_train)
	filename = "model/Lasso_model_%d.pkl" %(int(item_nbr))
	joblib.dump(clf, filename)

def lasso_test(X_test, item_nbr):
	scalar_name = "scalar/scale_%d.pkl" %(int(item_nbr))
	scalar = joblib.load(scalar_name)
	X_test = scalar.transform(X_test)
	
	model_name = "model/Lasso_model_%d.pkl" %(int(item_nbr))
	clf = joblib.load(model_name)
	y_pred = clf.predict(X_test)
	return y_pred
#####################################################################################
#####################################################################################
def svm_train(X_train, y_train, item_nbr, svm_kernel):
	if svm_kernel == 'linear':
		clf = svm.SVR(kernel=svm_kernel, C=1.0, epsilon=0.2)
	elif svm_kernel == 'poly':
		clf = svm.SVR(kernel=svm_kernel, C=1.0, epsilon=0.2, gamma=0.1)
	elif svm_kernel == 'rbf':
		clf = svm.SVR(kernel=svm_kernel, C=1.0, epsilon=0.2, gamma=0.1)
	
	clf.fit(X_train, y_train)
	filename = "model/svm_model_%s_%d.pkl" %(svm_kernel, int(item_nbr))
	joblib.dump(clf, filename)

def svm_test(X_test, item_nbr, svm_kernel): # Needed to edit more
	scalar_name = "scalar/scale_%d.pkl" %(int(item_nbr))
	scalar = joblib.load(scalar_name)
	X_test = scalar.transform(X_test)

	model_name = "model/svm_model_%s_%d.pkl" %(svm_kernel, int(item_nbr))

	clf = joblib.load(model_name)
	y_pred = clf.predict(X_test)
	return y_pred


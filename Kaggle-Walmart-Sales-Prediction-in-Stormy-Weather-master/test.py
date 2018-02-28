''' Test function for new samples '''

import time
import numpy as np 
import pandas as pd
import data_preprocess
import prediction_model
from sklearn.metrics import mean_squared_error
import multiprocessing
start = time.time()

#####################################################################################
# Evaluate the performance of the training model
#####################################################################################
''' 
Use part of training data as the test data to evaluate the performance of the 
prediction evaluate_model
'''
def evaluate_model():
	item_list = data_preprocess.get_item_list()
	#item_nbr = item_list[20]
	X_test, y_test = data_preprocess.split_item_data(item_nbr, 'test')

	y_pred = prediction_model.linear_regression_test(X_test, item_nbr)
	rmsle_error = prediction_model.rmsle_metrics(y_test, y_pred)
	print rmsle_error
	#print mean_squared_error(y_test, y_pred)
	#for i in range(len(y_test)):
	#	print y_test[i], y_pred[i]
#####################################################################################
#evaluate_model()
#####################################################################################
# Read, match and write test data to speed up
#####################################################################################
def group_test_data():
	store_station_pairs = data_preprocess.match_store_station()
	weather_file = data_preprocess.read_processed_weather()
	test_file = pd.read_csv("Data/test.csv", sep=',')	

	# Group all test data based on item_nbr and thus build index on item_nbr
	ori_index = np.matrix(test_file.index.values).transpose()
	new_data = np.append(ori_index, test_file.values[:,0:2], axis=1)
	df = pd.DataFrame(new_data, index=test_file.values[:,2], columns=["ori_index", "date", "store_nbr"])
	
	item_list = data_preprocess.get_item_list()
	for item_id in item_list:
		#item_id = item_list[5]
		item_data = df.loc[item_id].values[0:5,:]
		#item_weather_data = np.zeros((item_data.shape[0], weather_file.values.shape[1]+2))
		item_weather_data = []

		for i in range(item_data.shape[0]):
			original_index = item_data[i,0]
			date = item_data[i,1]
			store_nbr = item_data[i,2]
			weather_data = data_preprocess.get_weather_data(date,store_nbr,store_station_pairs,weather_file)
			weather_data = weather_data.astype(float)
			id_str = "%d_%d_%s" %(store_nbr, item_id, date)
			weather_data = np.array([original_index, id_str]+list(weather_data))

			#item_weather_data[i,:] = weather_data
			item_weather_data.append(weather_data)

		item_weather_data = np.array(item_weather_data)
		columns = np.append(np.array(["ori_index", "id_str"]), weather_file.columns.values, axis=1)
		df2 = pd.DataFrame(item_weather_data, index=None, columns = columns) 

		test_filename = "Test_item_data/item_%d.csv" %(int(item_id))
		df2.to_csv(test_filename, sep=',', index=None, columns=None)

	#print df2
#####################################################################################
group_test_data()

#####################################################################################
# Test and predict for unknown-label samples
#####################################################################################
def model_test(weather_data, item_nbr):	
	#y_pred = prediction_model.linear_regression_test(weather_data, item_nbr)
	#y_pred = prediction_model.lasso_test(weather_data, item_nbr)
	svm_kernel = 'rbf'
	y_pred = prediction_model.svm_test(weather_data, item_nbr, svm_kernel)

	return y_pred

'''Read in the test data and predict the label value '''
def predict(tasks):
	item_list = tasks.get()
	#print task_list

	div = len(item_list)/multiprocessing.cpu_count()
	pid = item_list[0]/div
	#print task_list[0],div,pid
	prediction_result = []
	count = 1.0
	for i in range(len(item_list)):
		item_id = item_list[i]
		test_filename = "Test_item_data/item_%d.csv" %(int(item_id))
		testItem_file = pd.read_csv("Data/test.csv", sep=',')

		testItem_data = testItem_file.values[:,2:]		
		testItem_pred = model_test(weather_data, item_id)

		for y_pred in testItem_pred:
			if y_pred < 0: # prediction result sometimes is negative !
				y_pred = 0

		writeData_pred = np.append(testItem_file.values[:,0:2], testItem_pred, axis=1)
		df = pd.DataFrame(writeData_pred, index=None, columns=np.array(["ori_index", "id_str", "units"]))
		df.to_csv("Data/item_result/result_%d.csv" %(item_id), sep=',', index=None)
		
		# Output how much is complete for each processor
		percent_comp = ((i+1)*1.0/len(item_list))*100
		if percent_comp >= count*10:
			print "processor %d is %d" %(pid, percent_comp), "% complete"
			count += 1.0

	end = time.time()
	print "processor %d Test End..." %(pid)
	print "Running time: %f" %(end-start)



def main():
	print "Test Start..."
	data_preprocess.multi_process(predict)

#main()




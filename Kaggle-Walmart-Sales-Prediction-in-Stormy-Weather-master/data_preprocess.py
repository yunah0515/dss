
''' Explore the statistics of dataset and Preprocess the dataset '''

import time
import sys
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, KFold
import multiprocessing

###################################################################################################
# Read weather data and process weather data for missing values
# Write processed weather data to csv file
###################################################################################################
''' Check if a string is a number (int or float) '''
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
	return False

''' 
Process and impute weather data information. weather_data here has 18 columns(without station_nbr and date) 
variable: data: array with 17 columns (codesum is removed, and all converted to float values)
		columns names wit 19 elements
Output: Write porcessed weather data into csv file
'''
def process_weather():
	weather_file = pd.read_csv("Data/weather.csv", sep=',')
	weather_data = weather_file.drop('codesum', axis = 1) # Drop "codesum" column

	data = weather_data.values[:,2:] # Get rid of station_nbr and date
	# Fill in the missing values
	for col in range(data.shape[1]):
		for row in range(data.shape[0]):
			if not is_number(data[row, col]):
				if row == 0:
					i = 1
					while not is_number(data[row+i, col]):
						i += 1
					data[row, col] = data[row+i, col]
				else:
					data[row, col] = data[row-1, col]
			data[row, col] = float(data[row, col]) # Convert string to float value
	
	weather_processed = np.append(weather_file.values[:,0:2], data, axis = 1)
	df = pd.DataFrame(weather_processed, index=None, columns=weather_data.columns.values)
	df.to_csv("Data/weather_processed.csv", sep=',', index=None)
###################################################################################################
# Match training data and corresponding weather data
# Get the weather dataset for each item
# Write the item dataset to csv file
##################################################################################################
''' Create store_nbr index on station_nbr using DataFrame and index '''
def match_store_station():
	key_file = pd.read_csv("Data/key.csv", sep=',')
	df = pd.DataFrame(key_file.values[:,1], index = key_file.values[:,0])
	return df

''' 
Create (date, station_nbr) index on weather data using DataFrame and Multi-index 
Output: DataFrame with 17 columns of features and 2 indexs on (date, station_nbr)
'''
def read_processed_weather():
	weather_file = pd.read_csv("Data/weather_processed.csv", sep = ',')
	multi_index = [weather_file.values[:,1], weather_file.values[:,0]]
	df = pd.DataFrame(weather_file.values[:,2:], index=multi_index, columns=weather_file.columns.values[2:])
	return df

''' 
Given date and store_nbr, return the corresponding weather data features. Specific use for match new test samples 
Output: 1x17 vectors
'''
def get_weather_data(date, store_nbr, store_station_pairs, weather_file):
	station_nbr = store_station_pairs.loc[store_nbr].values[0]
	weather_data = weather_file.loc[date, station_nbr].values
	return weather_data

'''Get the list of unique item id in train file'''
def get_item_list():
	train_file = pd.read_csv("Data/train.csv", sep=',')
	item_list = list(set(list(train_file.values[:,2])))
	return item_list

'''
Given the item_nbr, retrieve all corresponding weather features
Input: item_nbr - item id
		num - percentage of the number of samples should be retrieved
Output: Array with 18 columns, which contains 17 features and 1 label (the last column is label)
'''
def get_item_data(item_nbr, percent, store_station_pairs, weather_file, train_file):
	start = time.time()
	train_num = train_file.values[:,1].shape[0]
	
	item_index = np.where(train_file.values[:,2] == item_nbr)
	item_index = item_index[0]
	num = int(item_index.shape[0]*(1-percent)) # Select according to the given percentage, need to be improved
	iter_list = range(num, item_index.shape[0])
	item_dataset = np.zeros((len(iter_list), weather_file.values.shape[1]+1))
	count = 0
	#print len(iter_list)
	for i in iter_list:
		index = item_index[i]
		#print count
		date = train_file.values[:,0][index]
		store_nbr = train_file.values[:,1][index]
		units = train_file.values[:,3][index]
		
		weather_data = get_weather_data(date, store_nbr, store_station_pairs, weather_file)

		item_dataset[count,:] =  np.append(weather_data, [units], axis=0)
		count += 1
	end = time.time()
	print "Running time for item %d: %f" %(int(item_nbr), (end-start))
	return item_dataset

def write_item_data(tasks):
	item_list = tasks.get()
	#print item_list
	store_station_pairs = match_store_station()
	weather_file = read_processed_weather() # index has 2 columns and values has 17 columns
	train_file = pd.read_csv("Data/train.csv", sep=',')

	#item_list = get_item_list()
	#item_nbr = item_list[0]
	for item_nbr in item_list:
		item_dataset = get_item_data(item_nbr, 0.1, store_station_pairs, weather_file, train_file) # Give percentage
		df = pd.DataFrame(item_dataset, index=None, columns=None)
		filename = "Item_data/item_%d.csv" %(int(item_nbr))
		df.to_csv(filename, sep=',', index=None, columns=None)

'''Multiprocessing for using all cpu cores'''
def multi_process(traget_func):
	processors = multiprocessing.cpu_count()

	myTasks = multiprocessing.Queue()
	item_list = get_item_list()
	temp_part = []
	div = len(item_list)/(processors-1)
	rem = len(item_list)%(processors-1)
	ind = 0
	while ind < div*(processors-1): 
		temp_part.append(item_list[ind:ind+div])
		ind = ind+div
	temp_part.append(item_list[(len(item_list)-rem):])

	for each in temp_part:
		myTasks.put(each)

	Workers = [multiprocessing.Process(target = traget_func, args =(myTasks,)) for i in range(processors)]

	#Workers[0].start()

	for each in Workers:
		each.start()

#multi_process(write_item_data)
##################################################################################################

# Read Item dataset for a given item_nbr
# Split the item dataset for training and test
##################################################################################################
def read_item_data(item_nbr):
	filename = "Item_data/item_%d.csv" %(int(item_nbr))
	item_dataset = pd.read_csv(filename, sep=',')

	item_data = item_dataset.values[:,0:-1]
	item_label = item_dataset.values[:,-1]
	return item_data, item_label

def split_item_data(item_nbr, type):
	item_data, item_label = read_item_data(item_nbr)
	X_train, X_test, y_train, y_test = train_test_split(item_data, item_label, test_size=0.25, random_state=42)
	if type == 'train':
		return X_train, y_train
	elif type == 'test':
		return X_test, y_test
##################################################################################################

# Store the full matched weather data into csv file
# Too long to run, no use for now
##################################################################################################
''' 
Get the training dataset and match with weather features
Generate the csv file contains training data and corresponding weather features
'''
def get_full_data():
	train_file = pd.read_csv("Data/train.csv", sep=',')

	store_station_pairs = match_store_station()
	weather_file = read_weather() # index has 2 columns and values has 18 columns
	train_num = train_file.values[:,1].shape[0]
 	item_dataset = [] # Should be 20 columns
	for ind in range(train_num):
		store_nbr = train_file.values[:,1][ind]
		date = train_file.values[:,0][ind]
		weather_data = get_weather_data(date, store_nbr, store_station_pairs, weather_file)
		weather_data = np.append(train_file.values[:,2:][ind], weather_data, axis = 1) #[item_nbr, units, weather_features]

		item_dataset.append(weather_data)

	item_dataset = np.array(item_dataset)
	train_columns = np.append(np.array(['item_nbr','units']), weather_file.columns.values, axis=1)
	train_df = pd.DataFrame(item_dataset, index=None, columns=train_columns) # Build item_nbr index on units and weather features
	train_df.to_csv("train_feature_matrix.csv", sep=',', index=None)
	return train_df

#get_full_data() 

''' Read in the full training data with corresponding weather features, 
	should have 19 columns, and the first column is the item_nbr '''

def read_full_data():
	train_features = pd.read_csv("train_feature_matrix.csv", sep=',')
	train_df = pd.DataFrame(train_features.values[:,1:], index=train_features.values[:,0], columns=train_features.columns.values[1:])

	print train_df

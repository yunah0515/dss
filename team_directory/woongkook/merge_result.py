''' Merge the final results into one file '''

import re
import glob
import numpy as np
import pandas as pd

test_file = pd.read_csv("Data/test.csv", sep=',')	
test_num = test_file.values.shape[0]

pred_result = np.zeros((test_num, 2))
filenames = glob.glob("Data/item_result/*")
for filename in filenames:
	result_temp = df.read_csv(filename, sep=',')
	for i in range(result_temp.values.shape[0]):
		ori_index = result_temp.values[i,0]
		pred_result[ori_index, :] = result_temp.values[i,1:]

df = pd.DataFrame(pred_result, index=None, columns = np.array(["id", "units"]))

df.to_csv("Data/prediction_result_final.csv", sep=',', index=None)
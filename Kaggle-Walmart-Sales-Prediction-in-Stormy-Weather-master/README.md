# Walmart-Recruiting_Sales-in-Stormy-Weather
Walmart Recuriting in Kaggle Competition: To predict how the sales of product will be affected by weather

I. Running Instructions:

	run "python train.py"
	run "python test.py"
	run "python merge_result.py"

II. File Descriptions:

	Note: I use multiprocessing in Python to speed up the computation, merge may be needed to get the final result

	1. data_preprocess.py

		run multi_process() function to write item training data into local disk
		item training data are saved in the folder "item_data_0.01"

	2. train.py

		run "python train.py" to train the prediction model. The model parameters are saved into local disk
		model parameters are saved in the folder "model/" and "scalar/"

	3. test.py

		run evaluate_model() function to test the prediction model based on portion of training data
		run main() function to predict the test data
		prediction for each item is saved in the folder "Data/item_result/"

	4. merge_result.py

		run "python merge_result.py" to merge the prediction result of each item
		merged prediction is saved in the folder "Data/"

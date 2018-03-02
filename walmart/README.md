# [Walmart Sales In Stormy Weather](https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather)

This is the code for a machine learning competition hosted on kaggle by Walmart Labs. Participants were asked to accurately
predict the sales of 111 potentially weather-sensitive products (like umbrellas, bread, and milk) around the time of 
major weather events at 45 of their retail locations.

The interesting thing about the data was that the features were dependent on each other. 
For example the fields were marked as:

* date - the day of sales or weather
* store_nbr - an id representing one of the 45 stores
* station_nbr - an id representing one of 20 weather stations
* item_nbr - an id representing one of the 111 products
* units - the quantity sold of an item on a given day

So the feature vector was `[date,store_nbr,station_nbr,item_nbr,units]`. As this formed a tree like structure where each
feature value was a result of previous feature value, so intuitively using decision tree was a good choice. But decision 
trees have a bad habbit of overfitting so using Random Forest for improving overall efficiency was a good choice. 
 
I was able to fit the model at decent accuracy and it was performing well on validation and test set. At that time I
was at top 25 percent on the Leader board. However it was becoming harder to keep up. Anyways when 
the competition ended and the model was tested on the full test data, my model performed in top 28 percent. I was able to get a Root mean square error of 0.11355. 

The biggest mistake that I made was that even though I extracted the data according to the date when there was a weather event,
in spirit of competition I should have simply removed the data upto the date from where the test data started.
The winner of the competition did the same, even though this is a not good practice.

I had a lot of fun participating in this competition. It taught me a lot about data extraction, relevance of ensemble learning methods and boosting.




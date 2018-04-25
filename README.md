##### Fast Campus Data Science School 7th Team Project 1 Regression Analysis
# [Walmart Recruiting II: Sales in Stormy Weather](https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather)


![](https://github.com/yunah0515/dss7_Crawling_walmart/blob/master/image/kaggle%20main%20image.png?raw=true)

#### [Click for Project Report](https://github.com/yunah0515/dss7_Crawling_walmart/blob/master/main_presentation/1%ED%8C%80(Crawling)_B_%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C.ipynb)

# [ Overview ]

### (1) Team : Crawling
> - TaeHyoung Kim (s0dkhim)
> - YunAh Baek (yunah0515)
> - WoongKooK Seo (woongkook)

### (2) Dataset : 
> #### Walmart Sales Data 

### (3) Objective : 
> #### Predict how sales of weather-sensitive products are affected by snow and rain

<br>

# [Data Description]

> #### train : 4617600 rows, 2 columns
> - sales data for all stores & dates in the training set

> #### test : 526917 rows, 2 columns
> - stores & dates for forecasting (missing 'units', which you must predict) 

> #### key : 45 rows, 2 columns
> - the relational mapping between stores and the weather stations that cover them

> #### weather : 20517 rows, 20 columns
> - a file containing the NOAA weather information for each station and day


| Index | Feature     | Description                                                         |
|-------|:-------------:|:---------------------------------------------------------------------:|
| 1     | units       | the quantity sold of an item on a given day (Target)                |
| 2     | date        | the day of sales or weather                                         |
| 3     | store_nbr   | an id representing one of the 45 stores                             |
| 4     | station_nbr | an id representing one of 20 weather stations                       |
| 5     | item_nbr    | an id representing one of the 111 products                          |
| 6     | tmax        | maximum degrees Fahrenheit                                          |
| 7     | tmin        | minimum degrees Fahrenheit                                          |
| 8     | tavg        | average degrees Fahrenheit                                          |
| 9     | depart      | departure from normal                                               |
| 10    | dewpoint    | average dew point                                                   |
| 11    | wetbulb     | average wet bulb                                                    |
| 12    | heat        | heating (season begins with July)                                   |
| 13    | cool        | cooling (season begins with January)                                |
| 14    | sunrise     | sunrise (calculated, not observed)                                  |
| 15    | sunset      | sunset (calculated, not observed)                                   |
| 16    | codesum     | significant weather types (weather phenomena)                       |
| 17    | snowfall    | snowfall (inches an tenths) T = Trace M = Missing data              |
| 18    | preciptotal | water equivalent (inches and hundredths) T = Trace M = Missing data |
| 19    | stnpressure | average station pressure                                            |
| 20    | sealevel    | average sea level pressure                                          |
| 21    | resultspeed | resultant wind speed                                                |
| 22    | resultdir   | resultant wind direction                                            |
| 23    | avgspeed    | average wind speed                                                  |

> #### File description

![image.png](https://github.com/yunah0515/dss7_Crawling_walmart/blob/master/image/file%20description.png?raw=true)

<br>


# [Evaluation]
> ![](https://github.com/yunah0515/dss7_Crawling_walmart/blob/master/image/evaluation.png?raw=true)

<br>

# [Contents]

### (1) Preprocessing & EDA
> - Missing values : Assigning with the most recent value
> - Excluding unit 0 
> - Weather table with codesum removing and missing data processing
> - Adding holiday and other variables
> - The closer the value is to zero, the less distortion
> - Normalization of target data

### (2) Feature Selection
> - Categorical variable analysis
> - Numerical variables analysis: select 9 out of 17
> - Multicollinearity
> - Selecting the most influential 9 numerical variables
> - VIF

### (3) Modeling
> - Modeling fuction
> - OLS (Ordinary Least Squares)
> - Modeling by each store : remove outliers

### (4) Results

### (5) Kaggle Submission
> - Total Teams : 485 teams 
> - Final Score : 0.51053
> - Leaderboard : 361 / 485 

### (6) Follow-up
> - New feature selection
> - Modeling
> - Score

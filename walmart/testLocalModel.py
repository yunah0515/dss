import csv
import math
import numpy as np
import six
import main as da

train= csv.reader(open(r'preparedData.csv'))
train= [[i for i in y ] for y in train]
train= [[float(i) for i in y ] for y in train]
actualValue = []
for i in range(1806350,len(train),10):
    actualValue.append(train[i][3])


print len(train[0])
da.delCollumn(train,5)
da.delCollumn(train,4)
da.delCollumn(train,3)
print len(actualValue)


x = csv.reader(open(r'testData.csv'))
x = [[i for i in y ] for y in x]
x = [[float(i) for i in y ] for y in x]
o = da.getCollumn(x,3)
da.delCollumn(x,3)
print "data uploaded"

#from sklearn.tree import DecisionTreeRegressor
#clf = DecisionTreeRegressor(max_depth = 38)
#16:18=45,14=48,22=0.075,24=0,067,26=0.06,28=0.056,32=0.053,34=0,05,40=0.049
#from sklearn.svm import SVR
#clf= SVR(kernel= 'rbf', C = 1e3)
#from sklearn import linear_model
#clf = linear_model.LinearRegression()
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(random_state=0, n_estimators=60,max_depth = 38)


clf.fit(x,o)

print ' trained'
y = []
for i in range(1806350,len(train),10):
    y.append(max(0,clf.predict(train[i])))
print 'predict'
print len(y)
error = 0
for i in range (len(y)):

    error = error + (math.log(y[i] + 1)  - math.log(actualValue [i] +1))  ** 2

error = error/ float(len(y))
error = math.sqrt(error)
print  error





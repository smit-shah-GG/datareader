'''

THIS FINALLY WORKS!!!!!

Already logged (2011 - 2017):
>> Apple
>> Google
>> Microsoft
>> Intel
>> Netflix
>> Amazon
>> AMD
>> Disney
>> Meta
>> Adobe
>> General electric
>> Tesla (up till 2020)
>> Bitcoin in USD

'''
#Dataframes
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import math
import numpy as np
plt.style.use('dark_background')

#Models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

#Preprocessing
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

#Warning Control
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2020, 1, 11)

df = web.DataReader("BTC-USD", 'yahoo', start, end)
df.tail()

print("\n New execution:  ")
print("\n Bitcoin in USD: \n")
print(df)

close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

mavg.plot(label='mavg')
plt.legend()

rets = close_px / close_px.shift(1) - 1

dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

#K Nearest Neighbours Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

print("\n")
confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)

print(confidencereg)
print(confidencepoly2)
print(confidencepoly3)
print(confidenceknn)

forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan

last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

#Plotting the graph
for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.savefig('BTC-USD.png', facecolor='black', transparent=False)

print("\n Forecast: ")
print(dfreg['Forecast'])

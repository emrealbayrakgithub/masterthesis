from formulas import *
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

keras = tf.keras

#run command: python3 arima.py

#change the folder parameter accordingly
folder='/home/user/files/'

asels = pd.read_csv(folder + '15.csv')

print("bef")
print(asels.head())
print("bef")

# Change date column to datetime type
asels['date'] = pd.to_datetime(asels['date'], format="%d.%m.%Y")

asels.drop(["date_time", "time", "Stock"], axis=1, inplace=True)

print("aft")
print(asels.head())
print("aft")
############

series = asels['Open']

# Create train data set
train_split_date = '2018-02-05'
train_split_index = np.where(asels.date == train_split_date)[0][0]
print("index1")
print(train_split_index)

x_train = asels.loc[asels['date'] <= train_split_date]['Open']
print(x_train)


# Create test data set
test_split_date = '2019-02-01'
test_split_index = np.where(asels.date == test_split_date)[0][0]
x_test = asels.loc[asels['date'] >= test_split_date]['Open']
print("index2")
print(test_split_index)
print(x_test)

# Create valid data set
valid_split_index = (train_split_index.max(),test_split_index.min())
x_valid = asels.loc[(asels['date'] < test_split_date) & (asels['date'] > train_split_date)]['Open']
print("index3")
print(valid_split_index)
print(x_valid)
############

# set style of charts
sns.set(style="darkgrid")
plt.rcParams['figure.figsize'] = [10, 10]

# Create a plot showing the split of the train, valid, and test data
plt.plot(x_train, label = 'Train')
plt.plot(x_valid, label = 'Validate')
plt.plot(x_test, label = 'Test')
plt.title('Train Valid Test Split of Data')
plt.ylabel('Prices')
plt.xlabel('Timestep in Hours')
plt.legend()
plt.savefig(folder + "fig1.png")
print(x_train.index.max(),x_valid.index.min(),x_valid.index.max(),x_test.index.min(),x_test.index.max())

############

def test_stationarity(timeseries, figname, window = 12, cutoff = 0.01 ):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    #plt.show()
    figname = folder + figname + '.png'
    plt.savefig( figname)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag = 20 )
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)

    print(dfoutput)

############

test_stationarity(series, "fig2")

############

asels_close_diff_1 = series.diff()
asels_close_diff_1.dropna(inplace=True)
test_stationarity(asels_close_diff_1, "fig3")

############

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(asels_close_diff_1)
plt.xlabel('Lags (Hours)')
#plt.show()
plt.savefig(folder + "fig4.png")

## Plot of partial autocorrelation

plot_pacf(asels_close_diff_1)
plt.xlabel('Lags (Hours)')
#plt.show()
plt.savefig(folder + "fig5.png")

############

from statsmodels.tsa.arima_model import ARIMA

## fitting the model

asels_arima = ARIMA(x_train, order=(4,2,0))
asels_arima_fit = asels_arima.fit(disp=0)
print(asels_arima_fit.summary())


############

from scipy import stats
import statsmodels.api as sm
from scipy.stats import normaltest

residuals = asels_arima_fit.resid
print(normaltest(residuals))
if normaltest(residuals)[1] < .05:
    print('This distribution is not a normal distribution')
# returns a 2-tuple of the chi-squared statistic, and the associated p-value. the p-value is very small, meaning
# the residual is not a normal distribution

fig = plt.figure(figsize=(12,8))
ax0 = fig.add_subplot(111)

sns.distplot(residuals ,fit = stats.norm, ax = ax0) # need to import scipy.stats



# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(residuals)

#Now plot the distribution using 
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Residual distribution')

plt.savefig(folder + "fig6.png")

# ACF and PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(residuals, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(residuals, lags=40, ax=ax2)

plt.savefig(folder + "fig7.png")
############

plt.clf()

history = [x for x in x_train]

# establish list for predictions
model_predictions = []

# Count number of test data points
N_test_observations = len(x_test)

# loop through every data point
for time_point in list(x_test.index):
    model = ARIMA(history, order=(4,2,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = x_test[time_point]
    history.append(true_test_value)
MAE_error = keras.metrics.mean_absolute_error(x_test, model_predictions).numpy()
print('Testing Mean Squared Error is {}'.format(MAE_error))




plt.plot(x_test.index[-20:], model_predictions[-20:], color='blue',label='Predicted Price')
plt.plot(x_test.index[-20:], x_test[-20:], color='red', label='Actual Price')
plt.ylabel('Price')
plt.xlabel('Timestep in Hours')
plt.title('ARIMA(4,2,0) Forecast vs Actual')
# plt.xticks(np.arange(881,1259,50), df.Date[881:1259:50])
plt.legend()
plt.figure(figsize=(10,6))
plt.savefig(folder + "fig8.png")
plt.show()
############

plt.clf()

model_predictions = np.array(model_predictions).flatten()

# Calculate MAE
arima_mae = keras.metrics.mean_absolute_error(x_test, model_predictions).numpy()

# Save to our dictionary of model mae scores

arima_error = model_predictions - x_test

plt.plot(x_test.index, arima_error, color='blue',label='Error of Predictions')
plt.hlines(np.mean(arima_error),xmin=x_test.index.min(),xmax=x_test.index.max(), color = 'red', label = 'Mean Error')

plt.title('ASELS Prediction Error')
plt.xlabel('Timestep in Hours')
plt.ylabel('Error')
plt.legend()
plt.figure(figsize=(10,6))
plt.show()
plt.savefig(folder + "fig9.png")




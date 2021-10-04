import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GRU
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error
import time
import sys
from statistics import mean
from tensorflow.keras import losses
import decimal

### RUN COMMAND ###
# python3 gru.py 250 20 mse random_uniform 0.0001 30 adam 0.4
### RUN COMMAND ###


epoch_v = int(sys.argv[1])
batch_size_v = int(sys.argv[2])
loss_v = sys.argv[3]
kernel = sys.argv[4]
learningrate = float(sys.argv[5])
time_step_v = int(sys.argv[6])
optimizer_v = sys.argv[7]
momentsgd = float(sys.argv[8])

if loss_v == 'mse':
    lossf_v = losses.MeanSquaredError()
elif loss_v == 'mae':
    lossf_v = losses.MeanAbsoluteError()
elif loss_v == 'msle':
    lossf_v = losses.MeanSquaredLogarithmicError()
elif loss_v == 'logcosh':
    lossf_v = losses.LogCosh()

print("epoch: ", epoch_v)
print("batch_size: ", batch_size_v)

params = {
    "batch_size": batch_size_v,  # 20<16<10, 25 was a bust
    "epochs": epoch_v,
    "lr": learningrate,
    "time_steps": time_step_v
}

# change output_path accordingly
# output path is where the model will be saved. eg. /home/user/Models/
OUTPUT_PATH = '/home/user/Models/'

TIME_STEPS = params["time_steps"]
BATCH_SIZE = params["batch_size"]
LR = params["lr"]
EPOCHS = params["epochs"]

if optimizer_v == 'sgd':
    optimizer = optimizers.SGD(lr=LR, momentum=momentsgd)
elif optimizer_v == 'adam':
    optimizer = optimizers.Adam(lr=LR)
elif optimizer_v == 'rmsprop':
    optimizer = optimizers.RMSprop(lr=LR)

print("CONFIG:\n")
print(sys.argv[1] + " " + sys.argv[2] + " " + sys.argv[3] + " " + sys.argv[4] + " " + sys.argv[5] + " " + sys.argv[
    6] + " " + sys.argv[7] + " " + sys.argv[8])


# TIME_STEPS define how many units back in time you want your network to see. (3)
# BATCH_SIZE says how many samples of input do you want your Neural Net to see before updating the weights. (60)
# FEATURES is the number of attributes used to represent each time step. (100)

def write_to_file(filename, text):
    fh = open(filename, 'a')
    fh.write(text + "\n")
    fh.close()


def build_timeseries(mat, y_col_index):
    """
    Converts ndarray into timeseries format and supervised data format. Takes first TIME_STEPS
    number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.
    :param mat: ndarray which holds the dataset
    :param y_col_index: index of column which acts as output
    :return: returns two ndarrays-- input and output in format suitable to feed
    to GRU.
    """

    # y_col_index is the index of column that would act as output column in the train_cols list. Here the "Open" index is 0
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, y_col_index]
    print("length of time-series i/o", x.shape, y.shape)
    return x, y


def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0] % batch_size
    if (no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat


def create_model():
    # The GRU architecture
    regressorGRU = Sequential()
    # First GRU layer with Dropout regularisation
    regressorGRU.add(GRU(units=34, return_sequences=True, input_shape=(TIME_STEPS, x_t.shape[2]), activation='tanh',
                         kernel_initializer=kernel))
    regressorGRU.add(Dropout(0.2))
    # Second GRU layer
    regressorGRU.add(GRU(units=34, return_sequences=True, input_shape=(x_t.shape[2], 1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Fourth GRU layer
    regressorGRU.add(GRU(units=34, activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # The output layer
    regressorGRU.add(Dense(units=1))
    # Compiling the RNN
    regressorGRU.compile(optimizer=optimizer, loss=lossf_v, metrics=['mse', 'mae', 'msle', 'logcosh'])

    print("number of layers: ", len(regressorGRU.get_config()['layers']))


    # change this path accordingly
    csv_logger = CSVLogger(
        os.path.join(OUTPUT_PATH, 'gru2' + '.log'),
        append=True)

    history = regressorGRU.fit(x_t, y_t, epochs=EPOCHS, verbose=2, batch_size=BATCH_SIZE, shuffle=False,
                             validation_data=(trim_dataset(x_val, BATCH_SIZE), trim_dataset(y_val, BATCH_SIZE)),
                             callbacks=[csv_logger])

    gru_model_name = foldername + "/modelnew_impt5.h5"

    if not os.path.exists('Models/'):
        os.mkdir('Models/')

    if os.path.exists(gru_model_name):
        os.remove(gru_model_name)
    regressorGRU.save(gru_model_name)

    print(regressorGRU.get_config())
    return regressorGRU, history


if __name__ == '__main__':

    foldername = os.path.join(OUTPUT_PATH, "model_gru_ep_" + str(sys.argv[1]) + "_bs_" + str(sys.argv[2]) + "_loss_" + str(
        sys.argv[3]) + "_kernel_" + str(sys.argv[4]) + "_lr_" + str(
        sys.argv[5].replace('.', '')) + "_ts_" + str(sys.argv[6]) + "_opt_" + str(sys.argv[7]) + "_moment_" + str(
        sys.argv[8].replace('.', '')) + time.ctime())

    if not os.path.exists(foldername):
        os.mkdir(foldername)

    print(foldername)

    stime = time.time()

    # Read the data into a pandas dataframe
    df_ge = pd.read_csv("data_folder/15.csv", index_col="date_time")

    # Check any null values
    # print("checking if any null values are present\n", df_ge.isna().sum())
    train_cols = ["Open", "High", "Low", "Close", "Volume", "Sma2", "Sma3", "Sma4", "Sma5", "Sma6", "Sma7", "Rsi",
                  "Cci", "Up_move", "Down_move", "Dmp", "Dmm", "Adx", "Vpt", "Efi", "Wobv", "Vzo", "Pzo", "Tp", "Adl",
                  "Smma", "Tr", "Sar", "Vwap", "Ssma", "Dema", "Tema", "Trix", "Currency"]
    pred_colum_index = train_cols.index("Open")

    # Split the data into train and test parts
    df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)

    # Turn them into numpy arrays
    x_train = df_train.loc[:, train_cols].values
    x_test = df_test.loc[:, train_cols].values

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)

    x_test_max = x_test[:len(train_cols)].max(axis=0)
    x_train_max = x_train[:len(train_cols)].max(axis=0)

    #x_test_max = x_test[:1].max(axis=0)
    #x_train_max = x_train[:1].max(axis=0)
    #x_test_max = x_test_max[0]
    #x_train_max = x_train_max[0]
    print("x_test_max shape:", x_test_max)
    print("x_train_max:", x_train_max)

    print("x_test_max shape:", x_test_max.shape)
    print("x_train_max:", x_train_max.shape)

    # Short info about the dataset
    print("Train cols length:", len(train_cols))
    print("Train and Test size", len(df_train), len(df_test))
    print("Type of x_train: ", type(x_train))
    print("Type of x_test: ", type(x_test))
    print("Number of rows of x_train: ", x_train.shape[0])
    print("Number of columns of x_train: ", x_train.shape[1])
    print("Number of rows of x_test: ", x_test.shape[0])
    print("Number of columns of x_test: ", x_test.shape[1])

    # Build and trim the train dataset
    x_t, y_t = build_timeseries(x_train, pred_colum_index)
    print("x_t shape before trim: ", x_t.shape)
    print("y_t shape before trim: ", y_t.shape)

    x_t = trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)
    print("x_t shape after trim: ", x_t.shape)
    print("y_t shape after trim: ", y_t.shape)

    # Build and trim the test dataset
    x_temp, y_temp = build_timeseries(x_test, pred_colum_index)
    # splits both parts into two as the validation and test
    x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE), 2)
    y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE), 2)
    print("x_val shape: ", x_val.shape)
    print("x_test_t shape: ", x_test_t.shape)
    print("y_val shape: ", y_val.shape)
    print("y_test_t shape: ", y_test_t.shape)

    print("x_t.shape[2]: ", x_t.shape[2])
    # x_t.shape[2]:  34 -- number of features

    # Create the model
    model, history = create_model()

    print("HISTORY KEYS: ", history.history.keys())

    # Write loss values into info.txt file
    for key, value in history.history.items():
        print(key + ": " + str(mean(value)))
        write_to_file(os.path.join(foldername, 'info.txt'), '{0} : {1}'.format(key, decimal.Decimal(str(mean(value)))))

    # model.evaluate(x_test_t, y_test_t, batch_size=BATCH_SIZE

    # Make predictions. Model takes the test input data we created above and returns the predictions
    y_pred = model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
    y_pred = y_pred.flatten()
    # Trim the test output data
    y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
    # Calculate the error using the test output and predictions
    print("y_pred shape:", y_pred.shape)
    print("y_test_t shape:", y_test_t.shape)
    error = mean_squared_error(y_test_t, y_pred)
    print("Error is", error, y_pred.shape, y_test_t.shape)
    print(y_pred[0:15])
    print(y_test_t[0:15])

    write_to_file(os.path.join(foldername, 'info.txt'), '{0} : {1}'.format("Test Error:", decimal.Decimal(str(error))))
    write_to_file(os.path.join(foldername, 'info.txt'), '{0} : {1}'.format("Prediction:", str(y_pred[0:15])))
    write_to_file(os.path.join(foldername, 'info.txt'), '{0} : {1}'.format("Test:", str(y_test_t[0:15])))

    # convert the predicted value to range of real data
    #y_pred_org = (np.reshape(y_pred, (460, 1)) * x_test_max)
    #y_test_t_org = (np.reshape(y_test_t, (460, 1)) * x_test_max)
    y_pred_org = y_pred
    y_test_t_org = y_test_t
    print("y_pred_org shape:", y_pred_org.shape)
    print("y_test_t_org shape:", y_test_t_org.shape)
    #print(y_pred_org[0:15])
    #print(y_test_t_org[0:15])

    # Visualize the training data
    from matplotlib import pyplot as plt

    search_key = 'loss'
    reskey = [key for key in history.history.keys() if search_key not in key]

    reskey_not_val = [i for i in reskey if 'val' not in i]

    for i in range(len(reskey_not_val)):
        plt.figure()
        plt.ylim(0.000, 0.000009)
        plt.plot(history.history[reskey_not_val[i]])
        plt.plot(history.history[reskey[i]])
        plt.title(reskey_not_val[i])
        plt.ylabel(reskey_not_val[i])
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join(foldername, str(reskey_not_val[i]) + '.png'))
        plt.clf()


    # Visualize the prediction
    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(y_pred_org)
    plt.plot(y_test_t_org)
    plt.title('Prediction vs Real Stock Price')
    plt.ylabel('Price')
    plt.xlabel('Hours')
    plt.legend(['Prediction', 'Real'], loc='upper left')
    plt.savefig(os.path.join(foldername, 'pred_vs_real_' + time.ctime() + '.png'))
    print("program completed ", time.time() - stime // 60, "minutes : ", np.round(time.time() - stime % 60), "seconds")

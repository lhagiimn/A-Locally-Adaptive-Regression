from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import *
import math
from sklearn import datasets

def data_preprocessing(train, test, dep_var,
                       variables, conf_int):

    # selected variables
    training, val = train_test_split(train, test_size=0.15,
                                       random_state=1324)

    trainingY = np.expand_dims(np.asarray(training[dep_var]), axis=1)
    valY = np.expand_dims(np.asarray(val[dep_var]), axis=1)
    testY = np.expand_dims(np.asarray(test[dep_var]), axis=1)

    trainingX = np.asarray(training[variables])
    valX = np.asarray(val[variables])
    testX = np.asarray(test[variables])

    # train_sd_X = np.sqrt((training[variables] - train[variables].mean())**2)
    # val_sd_X = np.sqrt((val[variables] - train[variables].mean()) ** 2)
    # test_sd_X = np.sqrt((test[variables] - train[variables].mean()) ** 2)

    scaler = MinMaxScaler()
    scaler = scaler.fit(trainingX)
    scaled_trainX = scaler.transform(trainingX)
    scaled_valX = scaler.transform(valX)
    scaled_testX = scaler.transform(testX)

    #please make it comment
    # scaler_y = MinMaxScaler()
    # scaler_y = scaler_y.fit(trainingY)
    # trainingY = scaler_y.transform(trainingY)
    # valY = scaler_y.transform(valY)
    # testY = scaler_y.transform(testY)

    # confidence interval significance rate 0.025, 0.975
    coef_trainX = np.zeros([trainingX.shape[0], trainingX.shape[1]])
    std_trainX = np.zeros([trainingX.shape[0], trainingX.shape[1]])

    for i in range(coef_trainX.shape[0]):
        coef_trainX[i] = np.asarray(conf_int[0])
        std_trainX[i] = np.asarray(conf_int[1])

    coef_valX = np.zeros([valX.shape[0], valX.shape[1]])
    std_valX = np.zeros([valX.shape[0], valX.shape[1]])

    for i in range(coef_valX.shape[0]):
        coef_valX[i] = np.asarray(conf_int[0])
        std_valX[i] = np.asarray(conf_int[1])

    coef_testX = np.zeros([testX.shape[0], testX.shape[1]])
    std_testX = np.zeros([testX.shape[0], testX.shape[1]])

    for i in range(coef_testX.shape[0]):
        coef_testX[i] = np.asarray(conf_int[0])
        std_testX[i] = np.asarray(conf_int[1])

    #concatenate

    train_set = np.concatenate((trainingX, scaled_trainX, coef_trainX, std_trainX, trainingY), axis=1)
    val_set = np.concatenate((valX, scaled_valX, coef_valX, std_valX, valY), axis=1)
    test_set = np.concatenate((testX, scaled_testX, coef_testX, std_testX, testY), axis=1)

    #input data to torch array
    train_set = torch.from_numpy(train_set)
    val_set = torch.from_numpy(val_set)
    test_set = torch.from_numpy(test_set)

    return train_set, val_set, test_set



def evaluation(true_Y, pred_Y):
    rmse = math.sqrt(mean_squared_error(true_Y, pred_Y))
    mae = (mean_absolute_error(true_Y, pred_Y))
    smape = r2_score(true_Y, pred_Y)

    return rmse, mae, smape


def data_generation(n_samples, n_informative):
    X = []
    Y = []
    Coef = []

    i=0
    for num, n_inf in zip(n_samples, n_informative):
        x, y, coef = datasets.make_regression(n_samples=num, n_features=3,
                                              n_informative=n_inf, noise=3,
                                              coef=True, random_state=num)
        cl = np.expand_dims(np.zeros(len(x)) + i, axis=1)
        x = np.concatenate([x, cl], axis=1)
        X.append(x)
        Y.append(y)
        Coef.append(coef)

        i=i+1

    return X, Y, Coef


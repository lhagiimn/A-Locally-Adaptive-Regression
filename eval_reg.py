from Meta_Regression import *
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from Linear_regression import linear_regression
from utils import data_preprocessing, evaluation
import matplotlib.ticker as mtick
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")

def regress(trainX, trainY, test, dep_var, performance):

    methods = [LinearRegression]
    print(np.asarray(trainY))
    for method in methods:
        sk_linear_reg = method().fit(np.asarray(trainX), np.asarray(trainY))
        pred_reg_t = sk_linear_reg.predict(np.asarray(test.drop([dep_var], axis=1)))
        rmse, mae, smape = evaluation(test[dep_var], pred_reg_t)

        performance['Data'].append(data_name)
        performance['Method'].append('OLS')
        performance['Index'].append(k)
        performance['RMSE'].append(rmse)
        performance['MAE'].append(mae)
        performance['sMAPE'].append(smape)

    return performance


dep_vars= ['CO2']
data_names =['CO_2']


# Hyper-parameters
hidden_size = [16]#, 32, 64, 32, 10]
num_output = 1
num_epochs = 5000
learning_rate = 0.005
patience = 1000
batch_size = 16

k=0
for data_name, dep_var in zip(data_names, dep_vars):
    data = pd.read_csv('data/%s.csv' %data_name)
    data=data.dropna()
    data = data.set_index('NAME', drop=True)

    data_name='CO_3'

    # add intercept
    data['GNP'] = np.log(data['GNP'])
    #data['GNP^2'] = (np.square(data['GNP']))
    data['intercept'] = 1
    data[dep_var]=np.log(data[dep_var])
    performance = {'Data': [], 'Method': [], 'Index': [],
                   'RMSE': [], 'MAE': [], 'sMAPE': []}


    train = data.loc[data['origin']==0, ]
    test = data.loc[data['origin']==1, ]

    train = train.drop(['origin'], axis=1)
    test = test.drop(['origin'], axis=1)

    trainY = train[dep_var]
    #testY = test[dep_var]

    # print(np.mean(trainY), np.std(trainY))
    # print(np.mean(testY), np.std(testY))
    #
    # plt.style.use('seaborn-white')
    # kwargs = dict(histtype='stepfilled', alpha=0.5, density=True, bins=8, ec="k")
    #
    # plt.hist(trainY, **kwargs, label='2006: mean=0.73, sigma=1.53')
    # plt.hist(testY, **kwargs, label='2016: mean=0.49, sigma=1.76')
    # plt.xlabel('CO2 emission')
    # plt.ylabel('Probability density')
    # plt.title('Histogram of CO2 emission')
    # plt.legend()
    # plt.show()

    trainX = train.drop([dep_var], axis=1)

    #estimate linear regression
    linear_reg = linear_regression(train, dep_var)
    linear_model, sel_variables = linear_reg.regression()

    print(linear_model.summary())
    #exit('b')

    input_size = len(sel_variables)
    conf_int = linear_model.conf_int()

    coef = linear_model.params
    std = linear_model.bse
    conf_int[0] = coef
    conf_int[1] = std

    train_set, val_set, test_set = data_preprocessing(train=train, test=test,
                                                      variables=sel_variables,
                                                      conf_int=conf_int, dep_var=dep_var)


    meta_reg = MetaRegression(input_size=input_size,
                              hidden_size=hidden_size,
                              output_size=num_output)

    #print(summary(meta_reg, [(1, 12), (1, 12), (1, 12), (1, 12)]))
    if os.path.isfile('models/checkpoint_%s_%s.pt' % (data_name, k)):
        meta_reg = training(meta_reg, train_set, val_set, epochs=num_epochs,
                            batch_size=batch_size, lr=learning_rate, data_name=data_name,
                            idx=k, patience=patience)
    else:
        meta_reg, train_loss, valid_loss  = training(meta_reg, train_set, val_set, epochs=num_epochs,
                            batch_size=batch_size, lr=learning_rate, data_name=data_name,
                            idx=k, patience=patience)

        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
        plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(valid_loss)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('reg/early_stop_%s_%s.png' %(data_name, k))
        plt.close()


    output, coeff, testY, p_value = predict(meta_reg, test_set)

    output = output.detach().numpy()
    coeff = coeff.detach().numpy()
    testY = testY.detach().numpy()
    p_value = p_value.detach().numpy()

    # for i in range(2):
    #     plt.scatter(np.abs(np.asarray(testY)-np.asarray(output)),
    #                 p_value[:,i], c='r', alpha=0.5)
    #     plt.show()


    coef_pd = pd.DataFrame(data=coeff, columns=sel_variables)
    conc_out_pd = pd.DataFrame(data=p_value, columns=sel_variables)

    rmse, mae, smape = evaluation(testY, output)

    performance['Data'].append(data_name)
    performance['Method'].append('Meta_reg')
    performance['Index'].append(k)
    performance['RMSE'].append(rmse)
    performance['MAE'].append(mae)
    performance['sMAPE'].append(smape)

    performance = regress(trainX=trainX, trainY=trainY, test=test,
                          dep_var=dep_var, performance=performance)

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 12,
            }

    #test['GNP^2']=test['GNP']**2
    for var in sel_variables:
        other = test[sel_variables].drop([var], axis=1)
        for other_var in list(other):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(test[var], other[other_var], coef_pd[var], c='darkred',
                       marker='o')
            ax.set_xlabel(var, fontdict=font)
            ax.set_ylabel(other_var, fontdict=font)
            ax.set_zlabel('Effect of %s' %var, fontdict=font)
            ax.zaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            #plt.title('The relationship between CO2 and GNP')
            #plt.show()
            plt.savefig('reg/scatter_%s_%s_%s_%s.png' % (data_name, var, other_var, k))
            plt.close()

    res = pd.DataFrame.from_dict(performance)
    res.to_csv('reg/performance.csv', index=False, mode='a')

    coef_pd.index = test.index
    conc_out_pd.index = test.index
    res_df = pd.concat((test, coef_pd, conc_out_pd), axis=1)
    res_df.to_csv('reg/df_%s.csv' %data_name, index=True)













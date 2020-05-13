from Meta_regression_shared_weights import *
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from Linear_regression import linear_regression
from utils import data_preprocessing, evaluation
from torchsummary import summary

import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#dep_vars= ['PE', 'quality', 'v7']
#data_names =['ccpp', 'winequality_red', 'Yacht_Hydrodynamics']

# dep_vars= ['medv', 'v9', 'Y1']
# data_names =['BostonHousing', 'Concrete_Data', 'ENB2012_data']
#
# dep_vars= ['v9', 'v17']
# data_names =['kin8nm', 'Naval_Pro']

# dep_vars= ['RMSD']
# data_names =['CASP']

dep_vars= ['v9']
data_names =['kin8nm']

# dep_vars= ['v91']
# data_names =['MSD']

# Hyper-parameters
hidden_size = [64, 64, 16]
num_output = 1
num_epochs = 5000
learning_rate = 0.001
patience = 300

for data_name, dep_var in zip(data_names, dep_vars):
    data = pd.read_csv('data/%s.csv' %data_name)
    data=data.dropna()

    #data = data.drop(['v1'], axis=1)

    # add intercept
    data['intercept'] = 1
    if len(data[dep_var]) < 500:
        batch_size = 16
    elif len(data[dep_var]) < 1000:
        batch_size = 32
    elif len(data[dep_var]) < 5000:
        batch_size = 64
    elif len(data[dep_var]) < 10000:
        batch_size = 128
    elif len(data[dep_var]) < 20000:
        batch_size = 512
    else:
        batch_size = 10000

    performance = {'Data': [], 'Method': [], 'Index': [],
                   'RMSE': [], 'MAE': [], 'sMAPE': []}

    for k in range(2):
        train, test = train_test_split(data, test_size=0.1, random_state=k)

        trainY = train[dep_var]
        trainX = train.drop([dep_var], axis=1)

        #estimate linear regression
        linear_reg = linear_regression(train, dep_var)
        linear_model, sel_variables = linear_reg.regression()
        #print(linear_model.summary())

        input_size = len(sel_variables)
        #num_output = len(sel_variables)  #if you use shared neural network, please remove #

        conf_int = linear_model.conf_int()

        coef = linear_model.params
        std = linear_model.bse
        conf_int[0] = coef
        conf_int[1] = std

        print(data_name, len(sel_variables))
        exit('bye')
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
            plt.savefig('figures/early_stop_%s_%s.png' %(data_name, k))
            plt.close()


        output, coeff, testY, p_value = predict(meta_reg, test_set)

        output = output.detach().numpy()
        coeff = coeff.detach().numpy()
        testY = testY.detach().numpy()

        coef_pd = pd.DataFrame(data=coeff, columns=sel_variables)

        rmse, mae, smape = evaluation(testY, output)
        print(rmse, mae, smape)

        performance['Data'].append(data_name)
        performance['Method'].append('Meta_reg')
        performance['Index'].append(k)
        performance['RMSE'].append(rmse)
        performance['MAE'].append(mae)
        performance['sMAPE'].append(smape)

    '''
        for var in sel_variables:
            other = np.asarray(test[sel_variables].drop([var], axis=1))
            pca = PCA(n_components=1)
            pca.fit(other)
            low_dim = pca.transform(other)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(test[var], coef_pd[var], low_dim, c='r', marker='o')
            ax.set_xlabel('Data')
            ax.set_ylabel('Coeff')
            ax.set_zlabel('Low dim')
            plt.title('Scatter plot')
            plt.savefig('figures/scatter_%s_%s_%s.png' % (data_name, var, k))
            plt.close()
    '''

    res = pd.DataFrame.from_dict(performance)
    res.to_csv('models/performance_small.csv', index=False, mode='a')














import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.distributions.normal import Normal
import torchvision.models
from torchsummary import summary
import os
from scipy import stats

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# Fully connected neural network with one hidden layer

class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_output):
        super(NeuralNet, self).__init__()
        self.nn = torch.nn.ModuleList()
        for i in range(len(hidden_size)):
            if i==0:
                self.nn.append(nn.Linear(input_size, hidden_size[i]))
                self.nn.append(nn.ReLU())
            else:
                self.nn.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
                self.nn.append(nn.ReLU())

        self.nn.append(nn.Linear(hidden_size[-1], num_output))
        self.nn.append(nn.Sigmoid())

    def forward(self, x):

        global out
        for i in range(len(self.nn)):
            if i== 0:
                out = self.nn[0](x)
            else:
                out = self.nn[i](out)

        return out


class MetaRegression(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MetaRegression, self).__init__()
        self.coeff = torch.nn.ModuleList()

        for i in range(input_size):
            self.coeff.append(NeuralNet(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_output=output_size))

    def forward(self, X, norm_X, coef, std):

        output = []
        for i in range(norm_X.shape[1]):
            output.append(self.coeff[i](norm_X))

        p_value = (torch.cat(output, dim=1) + torch.tensor([0.000001]))/torch.tensor([1.00001])
        conc_out = m.icdf(p_value)

        coeff_out = torch.add(torch.mul(conc_out, std),  coef)

        #rewrite regression equation
        output = torch.sum(torch.mul(coeff_out, X), dim=1)

        return output, coeff_out, p_value


def training(meta_reg, train_data, val_data, epochs,
             data_name, idx, batch_size=8, lr=0.001, patience=500):

    if os.path.isfile('models/checkpoint_%s_%s.pt' % (data_name, idx)):

        # load the last checkpoint with the best model
        meta_reg.load_state_dict(torch.load('models/checkpoint_%s_%s.pt' % (data_name, idx)))

        return meta_reg

    else:

        split = int((train_data.shape[1] - 1) / 4)
        trainloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
        valloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(meta_reg.parameters(), lr=lr)

        train_losses = []
        valid_losses = []
        avg_train_losses = []
        avg_valid_losses = []
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        for e in range(epochs):

            meta_reg.train()
            for count, x in enumerate(trainloader):

                # Backward and optimize
                optimizer.zero_grad()
                trainX = x[:, :split]
                norm_trainX = x[:, split:2*split]
                coef = x[:, 2*split:3*split]
                std = x[:, 3*split:4*split]
                trainY = x[:, -1]

                # Forward pass
                output, coeff, conc_out = meta_reg(trainX.float(), norm_trainX.float(),
                                                   coef.float(), std.float())

                loss = criterion(output, trainY.float())
                loss.backward()

                #nn.utils.clip_grad_norm_(meta_reg.parameters(), 5)
                optimizer.step()
                train_losses.append(loss.item())

            meta_reg.eval()
            for val_count, val_x in enumerate(valloader):

                valX = val_x[:, :split]
                norm_valX = val_x[:, split:2 * split]
                coef_val = val_x[:, 2 * split:3 * split]
                std_val = val_x[:, 3 * split:4 * split]
                valY = val_x[:, -1]

                output_val, coeff_val, conc_out_val = meta_reg(valX.float(), norm_valX.float(),
                                                               coef_val.float(), std_val.float())

                val_loss = criterion(output_val, valY.float())
                valid_losses.append(val_loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(epochs))

            print_msg = (f'[{e:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            print(print_msg)

            early_stopping(valid_loss, meta_reg, path='models/checkpoint_%s_%s.pt' %(data_name, idx))

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        meta_reg.load_state_dict(torch.load('models/checkpoint_%s_%s.pt' %(data_name, idx)))

        return meta_reg, avg_train_losses, avg_valid_losses

def predict(meta_reg, test):

    split = int((test.shape[1] - 1) / 4)
    testX = test[:, :split]
    norm_testX = test[:, split:2 * split]
    coef_test = test[:, 2 * split:3 * split]
    std_test = test[:, 3 * split:4 * split]
    testY = test[:, -1]

    output, coeff, p_value = meta_reg(testX.float(), norm_testX.float(),
                                       coef_test.float(), std_test.float())

    return output, coeff, testY, p_value





















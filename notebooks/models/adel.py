


from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import random

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.fairness_utils import p_rule, DI

class ADEL(object):

    def __init__(self, learning_rate, batch_size, lamb,
                 num_epochs, NN_z, NN_y, NN_s,GPU, name="01"):
        self.lambda_ADV = lamb
        self.learning_rate = learning_rate
        self.batch_size=batch_size
        self.num_epochs = num_epochs
        self.device = torch.device(GPU if torch.cuda.is_available() else "cpu")

        self.m_NN_z = NN_z().to(self.device)
        self.m_NN_y = NN_y().to(self.device)
        self.m_NN_s = NN_s().to(self.device)
        #self.GPU = GPU

        self.name = "adel_" + str(lamb) + "_" + name

    def fit(self, X_train, y_train, S_train, X_test=None, y_test=None, S_test=None, plot_losses=False):
        batch_no = int(len(X_train) // self.batch_size) + 1
        
        self.optimizer_z = torch.optim.Adam(self.m_NN_z.parameters(), lr=self.learning_rate)
        self.optimizer_y = torch.optim.Adam(self.m_NN_y.parameters(), lr=self.learning_rate)
        self.optimizer_s = torch.optim.Adam(self.m_NN_s.parameters(), lr=self.learning_rate)

        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        
        loss_all, loss_sall, loss_yall = [], [], []
        
        for epoch in range(self.num_epochs):
            x_train, ytrain, strain = shuffle(X_train,np.expand_dims(y_train,axis = 1),np.expand_dims(S_train,axis = 1))
            # Mini batch learning
            epsilon=0.00000000000000001
            for i in range(batch_no):
                start = i * self.batch_size
                end = start + self.batch_size

                x_var = Variable(torch.FloatTensor(x_train[start:end])).to(self.device)
                y_var = Variable(torch.FloatTensor(ytrain[start:end])).to(self.device)
                s_var = Variable(torch.FloatTensor(strain[start:end])).to(self.device)

                # Forward + Backward + Optimize
                Ypred_var0 =  self.m_NN_y(self.m_NN_z(x_var)).detach()
                
                Z0 = self.m_NN_z(x_var).detach()

                for e in range(5):
                    self.optimizer_y.zero_grad()
                    Ypred_var = self.m_NN_y(Z0)
                    lossY = criterion(Ypred_var, y_var)
                    lossY.backward()
                    self.optimizer_y.step()
                for l in range(50):
                    self.optimizer_s.zero_grad()
                    Spred_var = self.m_NN_s(Z0)
                    lossS = criterion(Spred_var, s_var)
                    lossS.backward()
                    self.optimizer_s.step()

                self.optimizer_z.zero_grad()
                Ypred_var = self.m_NN_y(self.m_NN_z(x_var))
                Spred_var = self.m_NN_s(self.m_NN_z(x_var))
                lossS = criterion(Spred_var, s_var)
                lossY = criterion(Ypred_var, y_var)
                loss =  lossY
                loss_sall.append(lossS.item())
                loss_yall.append(lossY.item())
                loss_all.append(loss.item())
                if epoch >= 50:
                    loss =  - self.lambda_ADV *lossS + lossY 
                loss.backward()
                self.optimizer_z.step()
      
            if epoch % 5 == 0:
                y_pred2=torch.sigmoid(self.m_NN_y(self.m_NN_z(torch.FloatTensor(X_train).to(self.device)))).cpu().data.numpy()
                if X_test is not None:
                    y_pred2t= torch.sigmoid(self.m_NN_y(torch.FloatTensor(X_test).to(self.device))).cpu().data.numpy()
                    print('epoch', epoch, 'loss', loss.cpu().data.numpy(), 'lossS', lossS.cpu().data.numpy(), 'lossY', lossY.cpu().data.numpy(),'P-rule', p_rule(y_pred2,S_train),'ACC_train',accuracy_score(y_train, y_pred2>0.5),
                      'P-ruletest', p_rule(y_pred2t,S_test),'ACC_test',accuracy_score(y_test, y_pred2t>0.5))
                else: 
                    print('epoch', epoch, 'loss', loss.cpu().data.numpy(), 'lossS', lossS.cpu().data.numpy(), 'lossY', lossY.cpu().data.numpy(),'P-rule', p_rule(y_pred2,S_train),'ACC_train',accuracy_score(y_train, y_pred2>0.5))
                    
        if plot_losses:
            df_losses = pd.DataFrame({'loss':loss_all, 'loss_S':loss_sall, 'loss_Y':loss_yall})
            sns.lineplot(data=df_losses)
            plt.show()
                              
                         
    def predict(self, X, threshold=0.5):
        return (torch.sigmoid(self.m_NN_y(self.m_NN_z(torch.FloatTensor(X).to(self.device)))).cpu().data.numpy()>threshold).astype('int')
    
    def predict_proba(self, X):
        return torch.sigmoid(self.m_NN_y(self.m_NN_z(torch.FloatTensor(X).to(self.device)))).cpu().data.numpy()
                
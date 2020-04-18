# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 22:10:04 2020

@author: Qinliang Xue
"""

from readdata import LoadPandasData
from readdata import LoadTensorData
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

def TrainModel(model,criterion,optimizer,epoch,batch_size,traindata):
    dataloader = Data.DataLoader(dataset=traindata, batch_size=batch_size, shuffle=True)
    for e in range(epoch):
        for i,(x,y) in enumerate(dataloader):
            out = model(x)
            loss = criterion(out,y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch {}: ".format(e)+str(i)+" loss:{}".format(loss.data.item()))

def TestModel(model,criterion,batch_size,testdata):
    model.eval()
    eval_loss = 0
    eval_acc = 0
    dataloader = Data.DataLoader(dataset=testdata, batch_size=batch_size, shuffle=False)
    for (x,y) in dataloader:
        out = model(x)
        loss = criterion(out, y.long())
        eval_loss += loss.data.item()*y.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == y.long()).sum()
        eval_acc += num_correct.item()
    print("Loss:{} Acc:{}".format(eval_loss / (len(testdata)),eval_acc / (len(testdata))))
    
class MLPclassifer():
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim, learning_rate):
        self.epoch = 20
        self.batch_size = 20
        self.learning_rate = learning_rate
        self.model = MLP(in_dim, n_hidden_1, n_hidden_2, out_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
    def fit(self,data,detail=False):
        data = torch.from_numpy(data.values).float()
        data = data[:(len(data)//self.batch_size)*self.batch_size]
        data = data.reshape((-1,self.batch_size,data.shape[-1]))
        for e in range(self.epoch):
            for i,d in enumerate(data):
                x = d[:,1:]
                y = d[:,0]
                out = self.model(x)
                loss = self.criterion(out,y.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if detail == True:
                    print("epoch {}: ".format(e)+str(i)+" loss:{}".format(loss.data.item()))
        return self
                    
    def predict(self,data):
        data = torch.from_numpy(data.values).float()
        out = self.model(data)
        _, pred = torch.max(out, 1)
        return pred
    def predict_proba(self,data):
        if type(data) != np.ndarray:
            data = data.values
        data = torch.from_numpy(data).float()
        out = self.model(data)
        probs = F.softmax(out, dim=1)
        return probs
    def fit_predict(self,data,detail=False):
        self.fit(data,detail)
        out = self.model(data)
        _, pred = torch.max(out, 1)
        return pred
    def score(self,x,y):
        x = torch.from_numpy(x.values).float()
        y = torch.from_numpy(y.values).long()
        out = self.model(x)
        _, pred = torch.max(out, 1)
        num_correct = (pred == y).sum()
        return num_correct.item() / len(x)

def classify():
    path = "../data.csv"
    mlpmodel = MLPclassifer(1151,300,100,4,0.05)
    train,test = LoadPandasData(path)
    mlpmodel = mlpmodel.fit(train)
    return mlpmodel

if __name__ == "__main__":
    path = "../data.csv"
    mlpmodel = MLPclassifer(1151,300,100,4,0.02)
    train,test = LoadPandasData(path)
    ytrain = train['label']
    xtrain = train.drop('label',axis=1)
    ytest = test['label']
    xtest = test.drop('label',axis=1)
    mlpmodel = mlpmodel.fit(train,detail=True)
    out = mlpmodel.predict(xtest)
    print(mlpmodel.score(xtest,ytest))
#    epoch = 20
#    batch_size = 20
#    learning_rate = 0.05
#    traindata, testdata = LoadTensorData(path)
#    mlpmodel = MLP(1151,300,100,4)
#    criterion = nn.CrossEntropyLoss()
#    optimizer = optim.SGD(mlpmodel.parameters(), lr=learning_rate)
#    TrainModel(mlpmodel,criterion,optimizer,epoch,traindata)
#    TestModel(mlpmodel,criterion,testdata)
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:45:18 2020

@author: Qinliang Xue
"""
from readdata import LoadPandasData
from meachine_learning import vote
from mlp import classify
from scipy import stats
import numpy as np
import torch
import joblib

class blender():
    def __init__(self,*args):
        self.__models = args
    def predict(self,data):
        outs = [list(model.predict(data)) for model in self.__models]
        return stats.mode(outs)[0][0]
    def predict_proba(self,data):
        outs = [torch.Tensor(model.predict_proba(data)) for model in self.__models]
        Sum = torch.zeros(outs[0].shape)
        for o in outs:
            Sum += o
        return (Sum / len(outs)).detach().numpy().astype(float)

def train():
    voter1, voter2 = vote()
    voter3 = classify()
    blendvoter = blender(voter1,voter2,voter3)
    joblib.dump(blendvoter,"../model")
    return blendvoter

def test():
    model = joblib.load("../model")
    train,test = LoadPandasData("../data.csv")
    ytest = test['label']
    xtest = test.drop('label',axis=1)
    out = model.predict(xtest)
    print(model.predict_proba(xtest))
    print((ytest==out).sum()/len(ytest))
    
if __name__ == '__main__':
    train()
    test()
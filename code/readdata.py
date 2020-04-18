# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:35:11 2020

@author: Qinliang Xue
"""

import pandas as pd
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision import transforms

class GetTensorDataSet(Dataset):  # from torch.utils.data import Dataset
    def __init__(self,path,mode):
        self.all_data = []  # 存放所有的数据
        df = pd.read_csv(path)
        df = df.drop('ID',axis=1)
        df = shuffle(df)
        df['label'][df['label']==0] = 0
        df['label'][df['label']==1000] = 1
        df['label'][df['label']==1500] = 2
        df['label'][df['label']==2000] = 3
        col_names = df.columns.values.tolist()  # 获取列的名字
        Range = (0,0)
        if mode == 'train':
            Range = (0,int(0.8*len(df)))
        elif mode == 'test':
            Range = (int(0.8*len(df)),len(df))
        else:
            raise Exception("wrong mode except train and test")
        for i in range(*Range):  # 遍历每一行的数据
            self.all_data.append([df.iloc[i][col_names[1:]], df.iloc[i][col_names[0]]])  # 将一个样本和label为一组存放进去

    def __getitem__(self, index):
        return torch.FloatTensor(self.all_data[index][0]), self.all_data[index][1]

    def __len__(self):  # 返回所有样本的数目
        return len(self.all_data)
#  使用标准类来构造数据 加载数据
def LoadTensorData(path):
    traindata = GetTensorDataSet(path,'train')  # 实例化自己构建的数据集
    testdata = GetTensorDataSet(path,'test')
    return traindata, testdata
    
def LoadPandasData(path,noid=True):
    data = pd.read_csv(path)
    if noid:
        data = data.drop('ID',axis=1)
    data = shuffle(data)
    data['label'][data['label']==0] = 0
    data['label'][data['label']==1000] = 1
    data['label'][data['label']==1500] = 2
    data['label'][data['label']==2000] = 3
    train = data[:int(0.8*len(data))]
    test = data[int(0.8*len(data)):]
    return train, test
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:17:26 2020

@author: Qinliang Xue
"""

from readdata import LoadPandasData
import numpy as np
import pandas as pd
import lime.lime_tabular as tabular
import joblib

model = joblib.load("../model")
train,test = LoadPandasData("../data.csv",noid=False)
data = pd.concat([train,test])
IDindex = data['ID'] #带有index的ID，之后靠这个来通过ID对应index
ID = IDindex.values #所有ID的值，从这里选取ID
label = data['label']
feature = data.drop('label',axis=1).drop('ID',axis=1)
columnname = list(feature.columns)
feature = feature.values
idnumber = ID[0] #从ID中选取一个
explainer = tabular.LimeTabularExplainer(feature,feature_names=columnname,class_names=[0,1,2,3],discretize_continuous=True)
exp = explainer.explain_instance(feature[int(IDindex[IDindex==idnumber].index.values),:],
                                model.predict_proba,
                                num_features=5,
                                top_labels=5)
print(exp.predict_proba)
print(exp.as_list(label=0))
print(exp.as_list(label=1))
print(exp.as_list(label=2))
print(exp.as_list(label=3))
print(exp.as_pyplot_figure(label=0))
print(exp.as_pyplot_figure(label=1))
print(exp.as_pyplot_figure(label=2))
print(exp.as_pyplot_figure(label=3))
exp.save_to_file("../exp.html")
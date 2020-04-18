# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:17:26 2020

@author: Qinliang Xue
"""

from readdata import LoadPandasData
import numpy as np
import lime.lime_tabular as tabular
import joblib

model = joblib.load("../model")
train,test = LoadPandasData("../data.csv")
ytrain = train['label']
xtrain = train.drop('label',axis=1)
ytest = test['label']
xtest = test.drop('label',axis=1)
columnname = list(xtest.columns)
xtest = np.array(xtest)
explainer = tabular.LimeTabularExplainer(xtest,feature_names=columnname,class_names=[0,1,2,3],discretize_continuous=True)
exp = explainer.explain_instance(xtest[0],
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
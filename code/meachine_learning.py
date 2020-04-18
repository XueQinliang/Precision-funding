# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:11:56 2020

@author: Qinliang Xue
"""
from readdata import LoadPandasData
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

def kfold():
    kf = KFold(n_splits=5)
    train,test = LoadPandasData("../data.csv")
    ytrain = train['label']
    xtrain = train.drop('label',axis=1)
    ytest = test['label']
    xtest = test.drop('label',axis=1)
    alldata = pd.concat([train,test],axis=0,ignore_index=True)
    for train_index,test_index in kf.split(alldata):
        train = alldata.iloc[train_index]
        test = alldata.iloc[test_index]
        ytrain = train['label']
        xtrain = train.drop('label',axis=1)
        ytest = test['label']
        xtest = test.drop('label',axis=1)
        clf = DecisionTreeClassifier(random_state=0)
        rfc = RandomForestClassifier(random_state=0)
        gra = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
        knn = KNeighborsClassifier()
        lgt = LogisticRegression()
        clf = clf.fit(xtrain,ytrain)
        rfc = rfc.fit(xtrain,ytrain)
        lgt = lgt.fit(xtrain,ytrain)
        gra = gra.fit(xtrain,ytrain)
        knn = knn.fit(xtrain,ytrain)
        score_c = clf.score(xtest,ytest)
        score_r = rfc.score(xtest,ytest)
        score_l = lgt.score(xtest,ytest)
        score_g = gra.score(xtest,ytest)
        score_k = knn.score(xtest,ytest)
        print("Single Tree:{}".format(score_c),
              "Random Forest:{}".format(score_r),
              "GBDT:{}".format(score_g),
              "Logit Regression:{}".format(score_l),
              "K Neighbors:{}".format(score_k))

def vote(detail=False):
    train,test = LoadPandasData("../data.csv")
    ytrain = train['label']
    xtrain = train.drop('label',axis=1)
    ytest = test['label']
    xtest = test.drop('label',axis=1)
    clf = DecisionTreeClassifier(random_state=0)
    rfc = RandomForestClassifier(random_state=0)
    svc = svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovr',probability=True)
    gra = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    knn = KNeighborsClassifier()
    lgt = LogisticRegression()
    eclf1 = VotingClassifier(estimators=[('clf',clf),('knn',knn),('svc',svc)], voting='soft', weights=[2,2,1])
    eclf2 = VotingClassifier(estimators=[('rfc',rfc),('gra',gra),('lgt',lgt)], voting='soft', weights=[2,2,1])
    eclf1 = eclf1.fit(xtrain,ytrain)
    eclf2 = eclf2.fit(xtrain,ytrain)
    if detail:
        score_c = cross_val_score(clf, xtest, ytest, cv=5, scoring='accuracy')
        score_r = cross_val_score(rfc, xtest, ytest, cv=5, scoring='accuracy')
        score_s = cross_val_score(svc, xtest, ytest, cv=5, scoring='accuracy')
        score_l = cross_val_score(lgt, xtest, ytest, cv=5, scoring='accuracy')
        score_g = cross_val_score(gra, xtest, ytest, cv=5, scoring='accuracy')
        score_k = cross_val_score(knn, xtest, ytest, cv=5, scoring='accuracy')
        score_e1 = cross_val_score(eclf1, xtest, ytest, cv=5, scoring='accuracy')
        score_e2 = cross_val_score(eclf2, xtest, ytest, cv=5, scoring='accuracy')
        print("Single Tree:{}".format(score_c),
              "Random Forest:{}".format(score_r),
              "SVM:{}".format(score_s),
              "GBDT:{}".format(score_g),
              "Logit Regression:{}".format(score_l),
              "K Neighbors:{}".format(score_k),
              "voting:{}".format(score_e1),
              "voting:{}".format(score_e2))
    return eclf1, eclf2
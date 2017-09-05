#encoding = utf-8
'''
Created Time: Jul 4, 2017
Author:Wangqiqi 
 
This code is to compare the performance between random label and selective label with active learning based on idea from @afshin rahimi.
'''

import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from time import time
import numpy as np
import itertools
import copy
#from xgboost import XGBClassifier
import random
import sys
#import  matplotlib
#matplotlib.use('Agg') #used in Linux
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC,SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier,Perceptron,PassiveAggressiveClassifier,RidgeClassifier
from sklearn.utils import check_random_state,shuffle
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics,preprocessing,cross_validation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


NUM_QUESTIONS = 50#unlabeled samples each time
PLOT_RESULTS_TIME = False
PLOT_RESULTS_ERROR = True
ACTIVE = True
ENCODING = 'GBK'
#load data
train = pd.read_csv('C:/Users/qiqi/Desktop/AL/reduced_features_undersample.csv',encoding = ENCODING )
X_unlabeled = pd.read_csv('C:/Users/qiqi/Desktop/AL/unlabeled_undersample.csv',encoding = ENCODING )

y_unl = X_unlabeled['LB']
X_unlabeled_new = copy.deepcopy(X_unlabeled)
y = train['LB']
X = copy.deepcopy(train)
X.drop(['XH','LB'],1,inplace = True)
X_unlabeled_new.drop(['XH','LB'],1,inplace = True)
#scale the features set
scaler = StandardScaler()
X  = scaler.fit_transform(X)
scaler2 = StandardScaler()
X_unlabeled_new = scaler2.fit_transform(X_unlabeled_new)
X_unlabeled_new = pd.DataFrame(X_unlabeled_new)

cv = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=42)
for train_index, test_index in cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(y_train)
    flag = 0
    sum1 = 0
    batch_flag = 50#experiment times
    E_out_qs = []#results from query sample
    E_out_random = []#results from random sample
    query_num = []#sample size in each experiment
    X_train = pd.DataFrame(X_train)
    X_train4 = copy.deepcopy(X_train)
    y_train4 = copy.deepcopy(y_train)
    X_train_random1 = copy.deepcopy(X_train)
    y_train_random1 = copy.deepcopy(y_train)   
    while batch_flag:
        if batch_flag == 50:
            X_train_new = X_train4
            y_train_new = y_train4
            X_train_random_new = X_train_random1
            y_train_random_new = y_train_random1
        else:
            X_train_new = X_train_new
            y_train_new = y_train_new
            X_train_random_new = X_train_random_new
            y_train_random_new = y_train_random_new
        ###############################################################################
        # Benchmark classifiers
        def benchmark(X_train_p, y_train_p):
            print("Training: ")
            #C_range = np.logspace(-2, 10, 13)
            #gamma_range = np.logspace(-9, 3, 13)
            #C_range = [0.01,0.008]
            #gamma_range = [0.0001,0.00001]
            #param_grid = dict(gamma=gamma_range, C=C_range)
            #cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
            #for train_index, test_index in cv:
            #  X_train, X_test = X[train_index], X[test_index]
            #  y_train, y_test = y[train_index], y[test_index]
            #grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
            #s = grid.fit(X_train, y_train)
            #output the best score
            #bs = grid.best_score_
            #print("Best score: %0.3f" % bs)
            #print("Best parameters set:")
            # output the best paras
            #best_parameters = grid.best_estimator_.get_params()
            #for param_name in sorted(param_grid.keys()):
            #    print("\t%s: %r" % (param_name, best_parameters[param_name]))
            #clf = SVC(C= 0.01, gamma = 0.0001,kernel='rbf')
            clf = RandomForestClassifier(n_estimators=8,n_jobs=-1)
            '''clf = XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=200, \
                                  silent=True, objective='binary:logistic', nthread=-1, \
                                  gamma=0, min_child_weight=1, max_delta_step=0, \
                                  subsample=0, colsample_bytree=1, colsample_bylevel=0, \
                                  reg_alpha=1, reg_lambda=1, scale_pos_weight=1, \
                                  base_score=0.5, seed=3, missing=None)'''
            clf.fit(X_train_p, y_train_p)
            print("--------------------------------------------------")
            print(X_train.shape)
            print("------------------------feature importance:----------------")
            print(clf.feature_importances_)
            pred = clf.predict(X_test)
            score = metrics.f1_score(y_test, pred)
            accscore = metrics.accuracy_score(y_test, pred)
            print ("pred count is %d" %len(pred))
            print ('accuracy score:     %0.3f' % accscore)
            print("f1-score:   %0.3f" % score)      
            print("classification report:")
            print(metrics.classification_report(y_test, pred,target_names=['0','1']))
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))
            auc_val = metrics.roc_auc_score(y_test, pred)
            print('AUC:')
            print(auc_val)
            #accscore = svc.score(X_test, y_test) 
            #print('SVC test_score: {0:.3f}'.format(accscore))
            #compute absolute confidence for each unlabeled sample in each class
            confidences = np.abs( SVC(C= 0.01, gamma = 0.0001,kernel='rbf').fit(X_train_p, y_train_p).decision_function(X_unlabeled_new))
            #if(len(categories) > 2):
            #    confidences = np.average(confidences, axix=1)
            sorted_confidences = np.argsort(confidences)
            question_samples = []
            #select top k low confidence unlabeled samples
            low_confidence_samples = sorted_confidences[0:NUM_QUESTIONS]
            #select top k high confidence unlabeled samples
            #high_confidence_samples = sorted_confidences[-NUM_QUESTIONS:]
            question_samples.extend(low_confidence_samples.tolist())
            #question_samples.extend(high_confidence_samples.tolist())       
            return  accscore ,question_samples

        accscore ,question_samples = benchmark(X_train_new,y_train_new)
        accscore_rd ,question_samples_rd = benchmark(X_train_random_new,y_train_random_new)
        print("-------score-----------")
        print(accscore,accscore_rd)
        sum1 += len(question_samples)
        query_num.append(sum1)
        #error
        E_out_qs.append(1-accscore)
        E_out_random.append(1-accscore_rd)
         
        if ACTIVE:
            used_index = []
            for i in question_samples:
              if  i not in used_index:
                random_sam = random.sample(list(X_unlabeled_new.index),1)[0]
                if random_sam not in used_index:
                 X_train2 = X_unlabeled_new.iloc[i]
                 y_train2 = pd.Series(y_unl[i],index = [i])
                 #add a new feature sample from random set
                 X_train_random_new = pd.concat([X_train_new,pd.DataFrame(X_unlabeled_new.ix[[random_sam],:])])
                 #add a new feature sample from query set,substitute the human label for simplification
                 X_train_new = pd.concat([X_train_new,pd.DataFrame(X_train2).T])
                 #add a new target sample from random set
                 y_train2_random = pd.Series(y_unl[random_sam],index = [random_sam])
                 y_train_random_new = y_train_random_new.append(y_train2_random)
                 #add a new target sample from random set
                 y_train_new = y_train_new.append(y_train2)
                 used_index.extend([i,random_sam])
              else:
                continue
        else:
            break
        batch_flag -=   1

    if PLOT_RESULTS_ERROR: 
        plt.figure(figsize=(12,8))
        plt.plot(query_num, E_out_qs, 'g', label='qs Eout')
        plt.plot(query_num, E_out_random, 'k', label='random Eout')
        plt.xlabel('Number of Queries')
        plt.ylabel('Error')
        plt.title('Experiment Result')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=5)
        plt.savefig('qs_r1.png')
        plt.show()

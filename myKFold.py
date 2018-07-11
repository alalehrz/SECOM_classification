import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as stats
import sklearn as sk
from sklearn import tree
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from xgboost.sklearn import XGBClassifier
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier,\
                             RandomForestClassifier, RandomTreesEmbedding, BaggingClassifier
from sklearn.feature_selection import SelectFromModel, RFE, VarianceThreshold, SelectKBest, chi2, f_classif,\
                                      GenericUnivariateSelect, SelectFdr, SelectFwe, SelectFpr, SelectPercentile,\
                                      mutual_info_classif


# Where –1 corresponds to a pass and 1 corresponds to a fail and the data time stamp is for that specific test point.
data = pd.read_csv("secom.data.txt",  delimiter=' ', header=None, index_col=False)
target = pd.read_csv("secom_labels.data.txt", header=None, index_col=False, delimiter=' ', usecols=[0])
data_size, feature_size = data.shape

'''pre processing steps'''

data = data.dropna(axis=1, thresh=int(2*data_size/4))
data_size, feature_size = data.shape
data = data.dropna(axis=0, thresh=int(2*feature_size/4))
imputer = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0)
target = np.asarray(target).reshape((data_size,))
data = np.asarray(data)
low_Var = VarianceThreshold(0.1)
scaler = preprocessing.StandardScaler()
pca = PCA(n_components=10,svd_solver='full')

''' customized accuracy scores '''


def tp_rate(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]/(confusion_matrix(y_true, y_pred)[0, 0] +
                                                                            confusion_matrix(y_true, y_pred)[0, 1])


def tn_rate(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]/(confusion_matrix(y_true, y_pred)[1, 1] +
                                                                            confusion_matrix(y_true, y_pred)[1, 0])


def BER(y_true, y_pred): return 1-(0.5*(confusion_matrix(y_true, y_pred)[0, 0]/(confusion_matrix(y_true, y_pred)[0, 0] +
                                                                                confusion_matrix(y_true, y_pred)[0, 1]) +
                                        confusion_matrix(y_true, y_pred)[1, 1] / (
                                        confusion_matrix(y_true, y_pred)[1, 1] +
                                        confusion_matrix(y_true, y_pred)[1, 0])))


clf = XGBClassifier(booster="gblinear", n_estimators=500)
kf = KFold(n_splits=10, shuffle=True, random_state=0)
tp = np.zeros(10)
ber = np.zeros(10)
tn = np.zeros(10)

clf_R = RFE(estimator=svm.SVR(kernel='linear', C=1), n_features_to_select=40, step=1)
rfe = SelectFromModel(clf_R)
select_k = SelectKBest(k=40)                         # other option mutual_info_classif


i = 0
for train_index, test_index in kf.split(data):

        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        data_train = imputer.fit_transform(data_train)
        ros = RandomUnderSampler(random_state=1)      # alternative ros = RandomOverSampler(random_state=1)
        data_train, target_train = ros.fit_sample(data_train, target_train)
        rfe = rfe.fit(data_train, target_train)
        pipeline = Pipeline(steps=[('impute', imputer),
                                   ('var', low_Var),
                                   ('scale', scaler),
                                   ('FS', select_k),
                                   ('clf', c)])

        pipeline.fit(data_train, target_train)
        predict = pipeline.predict(data_test)
        ber[i] = BER(target_test, predict)
        tp[i] = tp_rate(target_test, predict)
        tn[i] = tn_rate(target_test, predict)
        i += 1

print("TP: %0.2f (+/- %0.2f)" % (np.mean(tp),
                                  np.std(tp) * 2))
print("TN: %0.2f (+/- %0.2f)" % (np.mean(tn),
                                      np.std(tn) * 2))
print("BER: %0.2f (+/- %0.2f)" % (np.mean(ber),
                                     np.std(ber) * 2))
#coding=utf-8
from sklearn import datasets
from sklearn import svm, model_selection
import numpy as np

"""
选择估计器和它们的参数
"""

digits=datasets.load_digits()
digits.images.shape
gammas = np.logspace(-6, -1, 10)
svc = svm.SVC()
clf = model_selection.GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas),n_jobs=-1)
clf.fit(digits.data[:1000], digits.target[:1000])
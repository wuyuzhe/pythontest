#coding=utf-8

from sklearn import linear_model,datasets
"""
线性模型：回归到稀疏
"""
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]



regr = linear_model.Lasso(alpha=.3)
regr.fit(diabetes_X_train, diabetes_y_train)
regr.coef_  # very sparse coefficients
regr.score(diabetes_X_test, diabetes_y_test)

lin = linear_model.LinearRegression()
lin.fit(diabetes_X_train, diabetes_y_train)
lin.score(diabetes_X_test, diabetes_y_test)
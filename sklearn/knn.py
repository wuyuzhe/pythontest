#coding=utf-8
#knn
from sklearn.datasets import load_iris
import numpy as np
from sklearn import neighbors

iris = load_iris()
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data , iris.target)
knn.predict([[0.1 , 0.2 , 0.3 , 0.4]])

perm = np.random.permutation( iris.target.size )
iris.data = iris.data[perm]
iris.target = iris.target[perm]
knn.fit(iris.data[:100] , iris.target[:100])
knn.score(iris.data[100:] , iris.target[100:])
neigh_dist,neigh_end = neighbors.KNeighborsRegressor(X)
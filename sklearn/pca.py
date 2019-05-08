#coding=utf-8
from sklearn.datasets import load_iris
from sklearn import decomposition

iris = load_iris()
pca = decomposition.PCA(n_components=2)
pca.fit(iris.data)
X = pca.transform(iris.data)
import pylab as pl
pl.scatter(X[:, 0], X[:, 1], c=iris.target)
pl.show()
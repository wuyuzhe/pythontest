#coding=utf-8
#对鸢尾花做分类，假设分为3类
import numpy as np
from sklearn import cluster , datasets
from scipy import misc
import matplotlib.pyplot as plt

iris = datasets.load_iris()
k_means = cluster.KMeans(k=3)
k_means.fit(iris.data)
print(k_means.labels_[::10])
print(iris.target[::10])



lena = misc.lena().astype(np.float32)
X = lena.reshape((-1, 1)) # We need an (n_sample, n_feature) array
k_means = cluster.KMeans(5)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
lena_compressed = np.choose(labels, values)
lena_compressed.shape = lena.shape
plt.gray()
plt.imshow(lena_compressed)
plt.show()
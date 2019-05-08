from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
iris.data.shape
iris.target.shape

np.unique(iris.target)

#数码数据集
digits=datasets.load_digits()
digits.images.shape

import pylab as pl

pl.imshow(digits.images[0] , cmap=pl.cm.gray_r)
pl.show()

data = digits.images.reshape((digits.images.shape[0] , -1))

#svm
from sklearn import svm
clf = svm.LinearSVC()
clf.fit(iris.data , iris.target) #learn from the data
clf.predict([[5.0, 3.6, 1.3, 0.25]])
clf.coef_   #存取模型的参数

#knn
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data , iris.target)
knn.predict([[0.1 , 0.2 , 0.3 , 0.4]])

perm = np.random.permutation( iris.target.size )
iris.data = iris.data[perm]
iris.target = iris.target[perm]
knn.fit(iris.data[:100] , iris.target[:100])
knn.score(iris.data[100:] , iris.target[100:])
neigh_dist,neigh_end = self.kneighbors(X)
###############################
from sklearn import svm

svc = svm.SVC(kernel = 'linear')
svc.fit(iris.data , iris.target)
#try
svc = svm.SVC(kernel='poly' , degree = 3)
svc = svm.SVC(kernel='rbf')
##############      聚类 kmeans               ################
#对鸢尾花做分类，假设分为3类
from sklearn import cluster , datasets

iris = datasets.load_iris()
k_means = cluster.KMeans(k=3)
k_means.fit(iris.data)
print k_means.labels_[::10]
print iris.target[::10]
#################################
from scipy import misc
lena = misc.lena().astype(np.float32)
X = lena.reshape((-1, 1)) # We need an (n_sample, n_feature) array
k_means = cluster.KMeans(5)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
lena_compressed = np.choose(labels, values)
lena_compressed.shape = lena.shape 
import matplotlib.pyplot as plt
plt.gray()
plt.imshow(lena_compressed)
plt.show() 

######    pca 降维 ###############################################
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(iris.data)
X = pca.transform(iris.data)
import pylab as pl
pl.scatter(X[:, 0], X[:, 1], c=iris.target)
pl.show()

############### face #############################################
import numpy as np
import pylab as pl from sklearn 
import cross_validation, datasets, decomposition, svm  

lfw_people = datasets.fetch_
lfw_people(min_faces_per_person=70, resize=0.4)
perm = np.random.permutation(lfw_people.target.size)
lfw_people.data = lfw_people.data[perm]
lfw_people.target = lfw_people.target[perm]
faces = np.reshape(lfw_people.data, (lfw_people.target.shape[0], -1))
train, test = iter(cross_validation.StratifiedKFold(lfw_people.target, k=4)).next()
X_train, X_test = faces[train], faces[test]
y_train, y_test = lfw_people.target[train], lfw_people.target[test]# ..
# .. dimension reduction ..
pca = decomposition.RandomizedPCA(n_components=150, whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_pca, y_train)# ..
# .. predict on new images ..
for i in range(10):    
   print lfw_people.target_names[clf.predict(X_test_pca[i])[0]]    
   _ = pl.imshow(X_test[i].reshape(50, 37), cmap=pl.cm.gray)    
   _ = raw_input()
   
   
################   线性模型：回归到稀疏
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

from sklearn import linear_model
regr = linear_model.Lasso(alpha=.3)
regr.fit(diabetes_X_train, diabetes_y_train)
regr.coef_ # very sparse coefficients
regr.score(diabetes_X_test, diabetes_y_test)
 
lin = linear_model.LinearRegression()
lin.fit(diabetes_X_train, diabetes_y_train)
lin.score(diabetes_X_test, diabetes_y_test) 

################# 选择估计器和它们的参数
from sklearn import svm, grid_search
gammas = np.logspace(-6, -1, 10)
svc = svm.SVC()
clf = grid_search.GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas),n_jobs=-1)
clf.fit(digits.data[:1000], digits.target[:1000]) 

#交叉验证估计器
from sklearn import linear_model, datasets
lasso = linear_model.LassoCV()
diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target
lasso.fit(X_diabetes, y_diabetes)
lasso.alpha
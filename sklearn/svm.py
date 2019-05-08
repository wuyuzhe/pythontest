#coding=utf-8
from sklearn.datasets import load_iris
from sklearn import datasets,svm
import numpy as np
import pylab as pl

"""
svm.LinearSVC
svm.LinearSVR
svm.NuSVC
svm.NuSVR
svm.SVC
svc = svm.SVC(kernel='poly' , degree = 3)
svc = svm.SVC(kernel='rbf')
svm.SVR
svm.OneClassSVM
"""

iris = load_iris()
iris.data.shape
iris.target.shape
np.unique(iris.target)
#数码数据集
digits=datasets.load_digits()
digits.images.shape

pl.imshow(digits.images[0] , cmap=pl.cm.gray_r)
pl.show()
data = digits.images.reshape((digits.images.shape[0] , -1))

#svm
clf = svm.LinearSVC()
clf.fit(iris.data , iris.target) #learn from the data
clf.predict([[5.0, 3.6, 1.3, 0.25]])
clf.coef_   #存取模型的参数


svc = svm.SVC(kernel = 'linear')
svc.fit(iris.data , iris.target)

############### face #############################################
import numpy as np
import pylab as pl
from sklearn  import model_selection, datasets, decomposition, svm


lfw_people = datasets.fetch_
lfw_people(min_faces_per_person=70, resize=0.4)
perm = np.random.permutation(lfw_people.target.size)
lfw_people.data = lfw_people.data[perm]
lfw_people.target = lfw_people.target[perm]
faces = np.reshape(lfw_people.data, (lfw_people.target.shape[0], -1))
train, test = iter(model_selection.StratifiedKFold(lfw_people.target, k=4)).next()
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
   print(lfw_people.target_names[clf.predict(X_test_pca[i])[0]])
   _ = pl.imshow(X_test[i].reshape(50, 37), cmap=pl.cm.gray)    
   _ = input()
   
   







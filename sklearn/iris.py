from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


iris = load_iris()

iris.data

iris.target
###########数据预处理

#区间缩放，返回值为缩放到[0,1]区间的值
StandardScaler().fit_transform(iris.data)

#归一化
from sklearn.preprocessing import Normalizer
Normalizer().fit_tranform(iris.data)

#二值化
from sklearn.preprocessing import Binarizer
Binarizer(threshold=3).fit_transform(iris.data)

#哑编码
from sklearn.preprocessing import OneHotEncoder
OneHotEncoder().fit_transform(iris.target.reshape((-1,1)))

#缺失值计算
from numpy import vstack,array,nan
from sklearn.preprocessing import Imputer
Imputer().fit_transform(vstack((array([nan,nan,nan,nan]) , iris.data)))

#数据变换

from sklearn.preprocessing import PolynomialFeatures
PolynomialFeatures().fit_transform(iris.data)

from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
FunctionTransformer(log1p).fit_transform(iris.data)

#####3特征选择

#3.1filter过滤发
#方差选择发
from sklearn.feature_selection import VarianceThreshold
VarianceThreshold(threshold=3).fit_transform(iris.data)

#相关系数法
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
SelectKBest(lambda X,Y: array(map(lambda x:pearsonr(x,Y),X.T)).T , k=2).fit_transform(iris.data ,iris.target)

#卡方检验
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
SelectKBest(chi2 , k=2).fit_transform(iris.data , iris.target)

#互信息法
from sklearn.feature_selection import SelectKBest
from minepy import MINE
def mic(x,y):
    m = MINE()
    m.compute_score(x,y)
    return (m.mic(),0.5)

SelectKBest(lambda X,Y: array(map(lambda x:mic(x,Y),X.T)).T ,k=2).fit_transform(iris.data , iris.target)

#3.2 wraper
#递归特征消除法
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
RFE(estimator=LogisticRegression() , n_features_to_select=2).fit_transform(iris.data , iris.target)
#3.3 embedded
#基于惩罚项,不全，需要根据l2惩罚项模型构建逻辑回归选取做均衡
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

SelectFromModel(LogisticRegression(penalty='l1',C=0.1)).fit_transform(iris.data , iris.target)
#基于树模型的特征选择法
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data , iris.target)

######4维度下降

#主成分分析PCA
from sklearn.decomposition import PCA

PCA(n_components=2).fit_transform(iris.data)
#线性判别分析LDA
from sklearn.lda import LDA
LDA(n_components=2).fit_transform(iris.data)







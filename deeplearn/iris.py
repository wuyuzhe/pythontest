from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


iris = load_iris()

iris.data

iris.target
###########����Ԥ����

#�������ţ�����ֵΪ���ŵ�[0,1]�����ֵ
StandardScaler().fit_transform(iris.data)

#��һ��
from sklearn.preprocessing import Normalizer
Normalizer().fit_tranform(iris.data)

#��ֵ��
from sklearn.preprocessing import Binarizer
Binarizer(threshold=3).fit_transform(iris.data)

#�Ʊ���
from sklearn.preprocessing import OneHotEncoder
OneHotEncoder().fit_transform(iris.target.reshape((-1,1)))

#ȱʧֵ����
from numpy import vstack,array,nan
from sklearn.preprocessing import Imputer
Imputer().fit_transform(vstack((array([nan,nan,nan,nan]) , iris.data)))

#���ݱ任

from sklearn.preprocessing import PolynomialFeatures
PolynomialFeatures().fit_transform(iris.data)

from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
FunctionTransformer(log1p).fit_transform(iris.data)

#####3����ѡ��

#3.1filter���˷�
#����ѡ��
from sklearn.feature_selection import VarianceThreshold
VarianceThreshold(threshold=3).fit_transform(iris.data)

#���ϵ����
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
SelectKBest(lambda X,Y: array(map(lambda x:pearsonr(x,Y),X.T)).T , k=2).fit_transform(iris.data ,iris.target)

#��������
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
SelectKBest(chi2 , k=2).fit_transform(iris.data , iris.target)

#����Ϣ��
from sklearn.feature_selection import SelectKBest
from minepy import MINE
def mic(x,y):
    m = MINE()
    m.compute_score(x,y)
    return (m.mic(),0.5)

SelectKBest(lambda X,Y: array(map(lambda x:mic(x,Y),X.T)).T ,k=2).fit_transform(iris.data , iris.target)

#3.2 wraper
#�ݹ�����������
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
RFE(estimator=LogisticRegression() , n_features_to_select=2).fit_transform(iris.data , iris.target)
#3.3 embedded
#���ڳͷ���,��ȫ����Ҫ����l2�ͷ���ģ�͹����߼��ع�ѡȡ������
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

SelectFromModel(LogisticRegression(penalty='l1',C=0.1)).fit_transform(iris.data , iris.target)
#������ģ�͵�����ѡ��
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data , iris.target)

######4ά���½�

#���ɷַ���PCA
from sklearn.decomposition import PCA

PCA(n_components=2).fit_transform(iris.data)
#�����б����LDA
from sklearn.lda import LDA
LDA(n_components=2).fit_transform(iris.data)







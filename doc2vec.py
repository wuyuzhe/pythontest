#!/usr/bin/python
import sys
import numpy as np
import gensim

from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from sklearn.cross_validation import train_test_split

LabeledSentence = gensim.models.doc2vec.LabeledSentence
def get_dataset():
	with open(pos_file,'r') as infile:
		pos_reviews = infile.readlines()
	with open(neg_file,'r') as infile:
		neg_reviews = infile.readlines()
	with open(unsup_file,'r') as infile:
		unsup_reviews = infile.readlines()
	
	y = np.concatenate((np.ones(len(pos_reviews)) , np.zeros(len(neg_reviews))))
	
	x_train , x_test,y_train,y_test = train_test_split(np.concatenate((pos_reviews , neg_reviews)) , y,test_size=0.2)
	
	def cleanText(corpus):
		punctuation = """.,?!:;(){}[]"""
		corpus = [z.lower().replace('\n','') for z in corpus]
		corpus = [z.replace('<br/>','') for z in corpus]
		
		for c in punctuation:
			corpus = [z.replace(c ,' %s '%c) for z in corpus]
		corpus = [z.split() for z in corpus]
		return corpus
	
	x_train = cleanText(x_train)
	x_test = cleanText(x_test)
	unsup_reviews = cleanText(unsup_reviews)
	
	def labelizeReviews(reviews , label_type):
		labelized = []
		for i,v in enumerate(reviews):
			label = '%s_%s'%(label_type , i)
			labelized.append(LabeledSentence(v,[label]))
		return labelized
	x_train = labelizeReviews(x_train , 'TRAIN')
	x_test = labelizedReviews(x_test , 'TEST')
	unsup_reviews = labelizeReviews(unsup_reviews , 'UNSUP')
	
	return x_train,x_test,unsup_reviews,y_train,y_test

#读取向量
def getVecs(model,corpus , size):
	vecs=[np.array(model.docvecs[z.tags[0]]).reshape((1,size)) for z in corpus]
	return np.concatenate(vecs)

def train(x_train,x_test ,unsup_reviews,size=400,epoch_num=10):
	#实例DM和DBOW模型
	model_dm = gensim.models.Doc2Vec(min_count=1,window=10,size=size,sample=le-3,negative=5,workers=3)
	model_dbow = gensim.models.Doc2Vec(min_count=1,window=10,size=size,sample=le-3,negative=5,dm=0,workers=3)
	
	#建立词典
	model_dm.build_vocab(np.concatenate((x_train,x_test,unsup_reviews)))
	model_dbow.build_vocab(np.concatenate((x_train,x_test,unsup_reviews)))
	
	all_train_reviews = np.concatenate((x_train , unsup_reviews))
	for epoch in range(epoch_num):
		perm = np.random.permutation(all_train_reviews.shape[0])
		model_dm.train(all_train_reviews[perm])
		model_dbow.train(all_train_reviews[perm])
	
	x_test = np.array(x_test)
	for epoch in range(epoch_num):
		perm = np.random.permutation(x_test.shape[0])
		model_dm.train(x_test[perm])
		model_dbow.train(x_test[perm])
		
	return model_dm.model_dbow

#将训练完成的数据转成vectors
def get_vectors(model_dm,model_dbow):

	train_vecs_dm = getVecs(model_dm,x_train,size)
	train_vecs_dbow = getVecs(model_dbow,x_train,size)
	train_vecs = np.hstack((train_vecs_dm , train_vecs_dbow))
	
	test_vecs_dm = getVecs(model_dm , x_test,size)
	test_vecs_dbow = getVecs(model_dbow , x_test,size)
	test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))
	
#使用分类器对文本向量进行分类训练
def Classifier(train_vecs,y_train , test_vecs,y_test):
	from sklearn.linear_model import SGDClassifier
	
	lr = SGDCLassifier(loss='log',penalty = 'll')
	lr.fit(train_vecs ,y_train)
	
	print 'Test Accuracy:%.2f'%lr.score(test_vecs,y_test)
	
	return lr

#绘制roc曲线，计算adc
def ROC_curve(lr,y_test):
	from sklearn.metrics import roc_curve,auc
	import matplotlib.pylot as plt
	
	pred_probas = lr.predict_proba(test_vecs)[:,1]
	
	fpr,tpr,_ = roc_curve(y_test,pred_probas)
	roc_auc = auc(fpr,tpr)
	plt.plot(fpr , tpr,label='area=%.2f' %roc_auc)
	plt.plot([0,1],[0,1],'k--')
	plt.xlim([0.0,1.1])
	plt.ylim([0.0 , 1.05])
	
	plt.show()
	
#
if __name__ == "__main__"
	size,epoch_num = 400,10
	x_train,x_test,unsup_reviews,y_train,y_test = get_dataset()
	model_dm,model_dbow = train(x_train,x_test ,unsup_reviews,size,epoch_num)
	train_vecs,test_vecs = get_vectors(model_dm,model_dbow)
	lr = Classifier(train_vecs , y_train,test_vecs,y_test)
	ROC_curve(lr ,y_test)
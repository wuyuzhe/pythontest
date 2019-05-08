import jieba
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

"""
需要很多相似非相似的句子来训练
"""
def getText():
    df_train = pd.read_csv('train_first.csv')
    discuss_train = list(df_train['Discuss'])
    return discuss_train

text = getText()
def cut_sentence(text):
    stop_list = [line[:-1] for line in open('stopwords.txt')]
    result = []
    for each in text:
        each_cut = jieba.cut(each)
        each_split = ' '.join(each_cut).strip()
        each_result = [word for word in each_split if word not in stop_list]
        result.append(' '.join(each_result))
    return result
b = cut_sentence(text)
TaggededDocument = Doc2Vec.TaggedDocument
def X_train(cut_sentence):
    x_train = []
    for i,text in enumerate(cut_sentence):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l-1] = word_list[l-1].strip()
        document = TaggededDocument(word_list , tags=[i])
        x_train.append(document)
    return x_train
c = X_train(b)

def train(x_train,size = 300):
    model = Doc2Vec(x_train , min_count=1,window=3,size=size,sample=le-3,nagative=5,workers=4)
    model.train(x_train,total_examples=model.copus_count,epochs=10)
    return model
model_dm = train(c)

str1 = u'不到 长城 非 好汉 爬 华山 长城 太 简单 值得 一去'
test_text = str1.split(' ')
infered_vector = model_dm.infer_vector(doc_words=test_text , alpha=0.025 , steps=500)
sims = model_dm.docvecs.most_similar([infered_vector] , topn=10)

print(sims)

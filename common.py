#coding=utf-8
from numpy import linalg
import jieba

#计算cos
def cos(a, b):
    num = float(a * b.T)
    denom = linalg.norm(a) * linalg.norm(b)
    result = num / denom
    return result

#读取文件进入列表
def read_file(file_input):
	f = open(file_input, 'r')
	data = []
	for line in f.readlines():
		line.strip('\n')
		data.append(line)
	f.close()
	return data

#jieba分词
def split_sentence(file_input):
	data = read_file(file_input)
	result = []
	for line in data:
        tmp = []
        words = jieba.cut(line)
        for word in words:
            tmp.append(word)
        result.append(tmp)
    return result
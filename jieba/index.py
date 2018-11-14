#coding=utf-8

import jieba
import jieba.posseg as pseg
import os

filepath = 'a.txt'
f = open(filepath , 'r')
str = ''

while True:
	line = f.readline()
	if not line:
		break
	str += ' '+line
f.close()

jieba.load_userdict('userdict.txt')
strings = str.split('。')
for string in strings:
	words = pseg.cut(string)
	print('&&&&&&&&&&&&&&&&&&&&&&&')
	for word,flag in words:
		print("%s %s"%(word,flag))


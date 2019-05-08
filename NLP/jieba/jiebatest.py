#coding=utf-8
import jieba

str = '可选择低预混胰岛素类似物、低预混人胰岛素、中预混胰岛素类似物或中预混人胰岛素。'
words = jieba.cut(str)

final = ""
#去除停用词
for word in words:
	#if word not in stopkey:
	final += " "+word
if final != ' \n':
    print(final)
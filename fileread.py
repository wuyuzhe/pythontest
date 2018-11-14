#!/usr/bin/env python
#coding=utf-8
f = open(path , "r" , encoding='utf-8')
f.write() 
f.read("filepath" , 'r' , encoding='utf-8')#对于3.x 以上的python得指明读取文件的编码格式
f.readline() 获取一行
f.writeline()  写一行
f.close()

os.rename()  重命名
os.remove()  移除
os.mkdir()/os.mkdirs()     创建目录/递归创建目录
os.chdir()      改变当前目录
os.rmdir()      删除目录

pickle（除了最早的版本外）是二进制格式的，所以你应该带 'b' 标志打开文件。


jieba.load_userdict("userdict.txt")
f = open("ymt.txt" , "r" , encoding='utf-8')
fw = open("result.txt", "w" , encoding='utf-8')
stopkey=[line.strip() for line in open('stop_words.txt' , 'r' , encoding='utf-8').readlines()]
#逐行处理
while True:
    line = f.readline()
    if not line:
        break
    words = jieba.cut(line)
    final = ""
    #去除停用词
    for word in words:
    	if word not in stopkey:
    		final += " "+word
    if final != ' \n':
    	fw.write(str(final))
f.close()
fw.close()
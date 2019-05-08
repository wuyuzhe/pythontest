#coding=utf-8

import html

"""
清洗检索记录
"""
f = open('C:/Users/wuyuzhe/Desktop/user_api_search_log.txt' , 'r+' , encoding='utf-8',errors='ignore')
f_w = open('f:/pywww/haodaifu/201701.txt' , 'a+' , encoding='utf-8')

cache = []
timenum = 0

for line in f.readlines():
    line = line.replace('"','')
    data = line.split('\t')
    if len(data) > 5 and data[4].isdigit():
        key = data[3]+data[4]
        if key in cache:
            continue
        if int(data[4]) > 1000000 and abs(int(data[4]) -timenum) > 2:
            cache = []
            timenum = int(data[4])
        cache.append(key)
        f_w.write(line)
f.close()
f_w.close()






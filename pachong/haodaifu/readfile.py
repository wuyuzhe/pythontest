#coding=utf-8

import os
# import pymysql
import re
import html

path = 'E:/data/'

f = open('f:/pywww/haodaifu/b.txt','a+' , encoding='utf-8')
for fpath in os.listdir(path):
    fpath = path + fpath
    matchobj = re.search('YiShengDetail' , fpath , re.M|re.I)
    if matchobj:
        findex = open(fpath , 'r' , encoding='gbk',errors='ignore')
        # db = pymysql.connect(host='127.0.0.1', user='root', passwd='123456', db='yisheng', charset='utf8')
        for line in findex.readlines():
            f.write(html.unescape(line))
        findex.close()
    else:
        continue
f.close()

# data = line.split('\t')
# cursor = db.cursor()
# db.execute("insert into docter (province,city,hospital,hospital_level,department_l,department_s,username,ptitle,goodat,visit,hassite,site,"
#            "hit,lasthit,doc,patients,lastpatients,wechat_patients,all_patients,vote,message,gift,lastlogin,created_at,recommend,satisfy,"
#            "help_patients,attitude_satify,2week_patients,site_ask,phone_ask,pre_reg,2year_vote,2yearago_vote,all_share,experience,message2,"
#            "gift2) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,)")



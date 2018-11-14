#!/usr/local/bin/python
#-*- coding: utf-8 -*-

import os;
#gui
import tkinter;

#time
import time;
ticks = time.time();

#mysql
import MySQLdb
db = MySQLdb.connect("localhost" , "user" , "password" , "dbname");
cursor = db.corsor()
cursor.execute("select version()");

#web socket
import sockets = socket.socket()
host = socket.gethostname()
port = 12345
s.bind((host,host))
s.listen(5)

#email
import smtplib
smtpObj = smtplib.SMTP([host[,port[,localhostname]]])
smtpObj.sendmail(from_addr,to_addr ,msg[,mail_options,rcpt_options])

import thead
tread.start_new_thread(functionname , functiondata)

#json
import json
data = [{'a':'1','b':'2'}]
json = json.dumps(data)
json.loads(jsondata)
#xml
import xml.sax
parser = xml.sax.make_parser()
parser.setFeature(xml.sax.handler.feature_namespaces , 0)
Handler = MovieHandler()
parser.setContentHandler(Handler)
parser.parse("movies.xml")
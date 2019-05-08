# -*- coding: utf-8 -*-
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

#load data
#another url http://cdn.powerxing.com/files/lr-binary.csv
#df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")
df = pd.read_csv("http://cdn.powerxing.com/files/lr-binary.csv")

print df.head()

df.columns = ["admit" ,"gre" , "gpa" , "prestige"]

print df.columns

#summarize the data
print df.describe()

#查看每一列的标准差
print df.std()

#频率表,表示prestige与admit的值相应的数量关系
print pd.crosstab(df['admit'] , df['prestige'] , rownames=['admit'])


#plot all of the columns
df.hist()
#pl.show()

#将prestige设为虚拟变量
dummy_ranks = pd.get_dummies(df['prestige'] , prefix='prestige')
print "\ndummy_ranks head\n"
print dummy_ranks.head()

#为逻辑回归创建所需的data frame
#除admit、gre、gpa外，加入了上面常见的虚拟变量
cols_to_keep = ['admit' , 'gre' , 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:,'prestige_2':])
print data.head()

#需要自行添加逻辑回归所需的intercept变量
data['intercept'] = 1.0

#执行逻辑回归
train_cols = data.columns[1:]
logit = sm.Logit(data['admit'] , data[train_cols])
result = logit.fit()

#coding=utf-8
import numpy as np
#np.frombufer

#创建数组
np.empty() #创建空数组
b = np.ones([2,2] , dtype=int) #默认为浮点数

#从已有的创建数组
x = [1,2,3]
x = (1,2,3)
a = np.asarray(x)

#从范围创建数组
np.arange(start=0,stop=10,step=2,dtype=int)
np.linespace() #创建一维数组,等差数列
np.logspace()  #创建一维数组,等比数列





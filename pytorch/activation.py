#coding=utf-8
'''
逻辑回归
神经网络
'''

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.linspace(-5 , 5 ,200)#做一些数据来观看图像
x = Variable(x)
x_np = x.data.numpy()

#常用激励函数
y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
y_softmax = F.softmax(x) #softmax比较特殊，不能直接显示，概率函数，用于分类

plt.figure(1, figsize=(8, 6))

plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)#设定画布位置
plt.plot(x_np, y_tanh, c='red', label='tanh')#绘制图形
plt.ylim((-1.2, 1.2))#设置y轴可读范围
plt.legend(loc='best')#设定图例显示位置

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()

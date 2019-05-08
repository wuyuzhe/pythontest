#coding = utf-8

import torch.nn as nn
import torch as t

lstm = nn.LSTM(input_size=4,#输入数据的特征向量数
               hidden_size=10,#输出的特征数是10
               batch_first=True)#使用batch_first数据维度表达方式，即(batch_size,序列长度，特征数目)
x = t.randn(3,5,4)
h0 = t.randn(1,3,10)
c0 = t.randn(1,3,10)
output = lstm(x,(h0,c0))

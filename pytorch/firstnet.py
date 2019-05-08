import torch
import matplotlib.pyplot as plt

"""
回归
"""

x = torch.unsqueeze(torch.linspace(-1,1,100) , dim=1)
y = x.pow(2)+0.2*torch.rand(x.size())

import torch.nn.functional as F #激励函数

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()

        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=1,n_hidden=10,n_output=1)

print(net)
optimizer = torch.optim.SGD(net.parameters(),lr=0.2)
loss_func = torch.nn.MSELoss()


plt.ion()
plt.show()
for i in range(200):
    prediction = net(x)

    loss = loss_func(prediction,y)  #计算两者误差

    optimizer.zero_grad()  #清空上一步的残余更新参数值
    loss.backward()  #误差反向传播
    optimizer.step() #将参数更新值施加到net的parameters上
    if i%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy() , prediction.data.numpy() ,'r-',lw=5)
        plt.text(0.5 ,0,'Loss=%.4f' % loss.data.numpy() , fontdict={'size':'20','color':'red'})
        plt.pause(0.1)

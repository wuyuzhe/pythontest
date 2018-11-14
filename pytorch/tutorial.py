#coding=utf-8

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2)+0.2*torch.rand(x.size())

x,y = Variable(x),Variable(y)


class Net(torch.nn.Module):
	def __init__(self,n_features, n_hidden,n_output):
		super(Net,self).__init__()
		self.hidden = torch.nn.Linear(n_features,n_hidden)
		self.predict = torch.nn.Linear(n_hidden,n_output)
		
	def forward(self , x):
		x = F.relu(self.hidden(x))
		x = self.predict(x)
		return x
		
net = Net(1,10,1)
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr=0.5)#优化参数，lr学习效率
loss_func = torch.nn.MSELoss() #均方差

plt.ion()  #something about plotting
for t in range(1000):
	prediction = net(x)   #input x and predict based on x
	
	loss = loss_func(prediction, y) #first parameter is output,second is target
	
	optimizer.zero_grad() #clear gradients for next train
	loss.backward()     #backpropagation ,compute gradients
	optimizer.step()  #apply gradients
	if t%5 == 0:
		plt.cla()
		plt.scatter(x.data.numpy(),y.data.numpy())#散点图
		plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
		plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
		plt.pause(0.1)

plt.ioff()
plt.show()
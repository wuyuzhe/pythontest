import numpy as np

dist = np.sqrt(numpy.sum(numpy.square(vec1-vec2)))
dist = np.linalg.norm(vec1-vec2)
#创建矩阵
A = np.mat("0 1 2;1 0 3;4 -3 8")
#计算逆矩阵
inv = np.linalg.inv(A)
#求解线性方程
B = np.mat("1 -2 1;0 2 -8;-4 5 9")
b = np.array([0,8,-9])
x = np.linalg.solve(B,b)
#点积
np.dot(B,x)
#求解特征值
np.linalg.eigvals(C)
#求解特征值和特征向量
c1,c2 = np.linalg.eig(C)
#奇异值分解
np.linalg.svd(D , full_matrices = False)
#广义逆矩阵
np.linalg.pinv(E)
#计算行列式
np.linalg.det(F)


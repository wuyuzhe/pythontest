import numpy as np

dist = np.sqrt(numpy.sum(numpy.square(vec1-vec2)))
dist = np.linalg.norm(vec1-vec2)
#��������
A = np.mat("0 1 2;1 0 3;4 -3 8")
#���������
inv = np.linalg.inv(A)
#������Է���
B = np.mat("1 -2 1;0 2 -8;-4 5 9")
b = np.array([0,8,-9])
x = np.linalg.solve(B,b)
#���
np.dot(B,x)
#�������ֵ
np.linalg.eigvals(C)
#�������ֵ����������
c1,c2 = np.linalg.eig(C)
#����ֵ�ֽ�
np.linalg.svd(D , full_matrices = False)
#���������
np.linalg.pinv(E)
#��������ʽ
np.linalg.det(F)


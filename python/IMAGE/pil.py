#coding=utf-8

from PIL import Image
from sklearn import decomposition
import numpy as np

im = Image.open('F:\pic\\t.jpg')
# print im.mode
# im.show()

#convert 将图像转为其他模式
im = np.array(im.convert('L'))
im2 = 255 - im
im3 = (100.0/255)*im + 100
im4 = 255.0*(im/255.0)**2

pca = decomposition.truncated_svd()
pca.fit(im4)
U = pca.transform(im4)
Sigma = pca.singular_values_
V = pca.components_

im5 = np.dot(np.dot(U , np.diag(Sigma)) , V)
pil_im = Image.fromarray(np.uint8(im5))
pil_im.show()

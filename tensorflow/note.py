import tensorflow as tf

#卷积操作（conv）、池化操作（pooling）、归一化、loss、分类操作、embedding、RNN、Evaluation。
tf.nn

#主要提供的高层的神经网络，主要和卷积相关的，个人感觉是对tf.nn的进一步封装，tf.nn会更底层一些。
tf.layers

#计算图中的网络层、正则化、摘要操作、是构建计算图的高级操作，但是tf.contrib包含不稳定和实验代码，有可能以后API会改变。
tf.contrib
tf.contrib.bayesflow
tf.contrib.bayesflow.entropy.renyi_alpha()
tf.contrib.layers

#DNN模型分类器
tf.estimator.DNNClassifier()
tf.estimator.DNNRegressor
tf.estimator.DNNLinearCombinedClassifier
tf.estimator.DNNLinearCombinedRegressor
####################################################################
tf.abs
tf.accumulate_n  #返回张量列表的元素和
tf.acos     #计算张量元素的acos
tf.add      #计算张量的和
tf.add_check_numerics_ops   #
tf.add_n    #添加张量元素
######################################################
#keras api
tf.keras
tf.linalg
tf.logging
tf.losses
tf.mainip
tf.math
tf.metrics
##################################################
#初始化
tf.initializers

#文件操作
tf.gfile
#位运算
tf.bitwise
#字符串转换
tf.compat
#操作张量图
tf.graph_util
#图片
tf.image


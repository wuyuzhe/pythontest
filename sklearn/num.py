import input_data
import tensorflow as tensorflow

mnist = input_data.read_data_sets("MNIST_data/" , one_hot=True)

#softmax回归模型
x = tf.placeholder("float" ,[None , 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#softmax
y = tf.nn.softmax(tf.matmul(x,W) + b)

#交叉熵
y_ = tf.placeholder("float" , [None , 10])

cross_entropy = tf.reduce_sum(y_*tf.log(y))
#reduce_sum 计算所有张量的和


train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化创建的变量
init = tf.initialize_all_variables()

#start model
sess = tf.Session
sess.run(init)

for i in range(1000):
	batch_xs,batch_ys = mnist.train.next_batch(100)
	sess.run(train_step , feed_dict={x:batch_xs , y_:batch_ys})


#评估
correct_prediction = tf.equal(tf.argmax(y,1) , tf.argmax(y_ , 1))

#tf.cast  类型转换
accuracy = tf.reduce_mean(tf.cast(correct_prediction , "float"))

print(sess.run(accuracy , feed_dict = {x:mnist.test.images,y_:mnist.test.labels}))
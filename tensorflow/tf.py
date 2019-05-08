#coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_MODE = 784
OUTPUT_MODE = 10

LAYER1_MODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor , avg_class , weights1 ,biases1 ,weights2,biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor , weights1) + biases1)
        return tf.matmul(layer1 , weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor , avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(tf.matmul(layer1 , avg_class.average(weights2)) + avg_class.average(biases2))

def train(mnist):
    x = tf.placeholder(tf.float32 , [None , INPUT_MODE] , name='x-input')
    y_ = tf.placeholder(tf.float32 , [None , OUTPUT_MODE] , name='y-input')

    weights1 = tf.Variable(tf.truncated_normal([INPUT_MODE , LAYER1_MODE] , stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1 , shape=[LAYER1_MODE]))

    weights2 = tf.Variable(tf.truncated_normal[LAYER1_MODE , OUTPUT_MODE] , stddev=0.1)
    biases2 = tf.Variable(tf.constant(0.1 , shape=[OUTPUT_MODE]))

    y = inference(x , None , weights1 , biases1 , weights2 , biases2)

    #训练轮数
    global_step = tf.Variable(0,trainable=False)

    #给定滑动平均衰减率和训练论述的变量，初始化滑动平均类。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY , global_step)

    #在所有代表神经网络参数的变量上使用滑动平均
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    #计算使用了滑动平均之后的钱箱传播结果
    average_y = inference(x , variable_averages , weights1 , biases1 ,weights2 , biases2)

    #交叉熵作为损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y , tf.argmax(y_ , 1))
    #计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE , global_step , mnist.train.num_examples /BATCH_SIZE , LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss , global_step=global_step)

    #下面两行代码与注释掉的代码等价，tensorflow提供了tf.group和tf.control_dependencies两种机制来更新参数和参数的滑动平均值
    #train_op = tf.group(train_step , variables_averages_op)
    with tf.control_dependencies([train_step , variables_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y , 1) , tf.argmax(y_ , 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x:mnist.validation.images ,
                         y_:mnist.validation.labels}

        test_feed = {x:mnist.test.images , y_:mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i%1000 == 0:
                validate_acc = sess.run(accuracy , feed_dict=validate_feed)
                print("After %d trainging step(s) validation, validation accuracy using average model is %g" % (i , validate_acc))

            xs , ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op , feed_dict={x:xs , y_:ys})

        test_acc = sess.run(accuracy , feed_dict=test_feed)
        print("After %d training step(s) , test accuracy using average model is %g" % (TRAINING_STEPS , test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data" , one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()



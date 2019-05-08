import tensorflow as tf

#input1 = tf.placeholder(tf.float32,[2,2]) #两行两列
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

<<<<<<< HEAD
output = tf.multiply(input1 , input2)
=======
output = tf.mul(input1 , input2)
>>>>>>> cf2f463e0d48f2a55c32beff341ca1b4b5c9e5ac
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[6.]}))
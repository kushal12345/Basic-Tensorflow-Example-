'''
BASIC TENSORFLOW EXAMPLES FOR BEGINEERS
Author: Kushal luitel
Project: https://github.com/kushal12345/Basic-Tensorflow-Example-
'''
import tensorflow as tf

#BASIC Constant Operations
a = tf.constant(5) #making a variable a and assigning a constant value
b = tf.constant(3) #making a variable b and assigning a constant value
#We need to make a tf.Session() for printing output
#These are basic Algebric Operation
with tf.Session() as sess:
    print ("a = 5 and b = 3")
    print ("Result of Addition of two constants(a + b) = ", sess.run(a+b))
    print ("Result of Subtraction of two constants(a - b) = ", sess.run(a-b))
    print ("Result of Multiplication of two constants(a * b) = ", sess.run(a*b))
    print ("Result of Division of two constants(a / b) = ", sess.run(a/b))

print ("=================================================================")
#A placeholder is simply a variable that we will assign data to at a later date.
#It allows us to create our operations and build our computation graph, without needing the data
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
#using Functions already made in Tensorflow we can do basic operation
add = tf.add(a,b)
sub = tf.subtract(a,b)
mul = tf.multiply(a,b)
div = tf.divide(a,b)
#making a Session to print
with tf.Session() as sess:
    print ("Addition with two variable = ",sess.run(add,feed_dict={a:5, b:3}))
    print ("Subtraction with two variable = ",sess.run(sub,feed_dict={a:5, b:3}))
    print ("Multiply with two variable = ",sess.run(mul,feed_dict={a:5, b:3}))
    print ("Divide with two variable = ",sess.run(div,feed_dict={a:5, b:3}))

print ("==================================================================")
#2D Matrix Multiplication
mat1 = tf.constant([1,2,3,4,5,6], shape=[2,3])
mat2 = tf.constant([7,8,9,10,11,12], shape=[3,2])

with tf.Session() as sess:
    print ("Matrix Multiplication is = ", tf.matmul(mat1, mat2))
#The output of the op is returned in 'result' as a numpy `ndarray` object.

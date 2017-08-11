import tensorflow as tf
import numpy as np
import math

sess = tf.InteractiveSession()

# Some simple constants
x1 = tf.Variable(tf.truncated_normal([5],
                 mean=3, stddev=1./math.sqrt(5)))
x2 = tf.Variable(tf.truncated_normal([5],
                 mean=-1, stddev=1./math.sqrt(5)))
x3 = tf.Variable(tf.truncated_normal([5],
                 mean=0, stddev=1./math.sqrt(5)))

sess.run(tf.global_variables_initializer())

# Squaring makes large values extreme (but positive)
# Be careful if you have negative values
sqx2 = x2 * x2
print(x2.eval())
print(sqx2.eval())

# Logarithm makes small values more pronounced (and negative)
# Be careful that your algorithm can handle negative numbers
logx1 = tf.log(x1)
print(x1.eval())
print(logx1.eval())

# "sigmoid" is a common transformation in deep learning
# Extreme values get flattened to +1 or 0
# Inputs closer to zero stay similar, sigmoid(0) = 0.5
sigx3 = tf.sigmoid(x3)
print(x3.eval())
print(sigx3.eval())

# We linearly combine multiple inputs, then transform
w1 = tf.constant(0.1)
w2 = tf.constant(0.2)
sess.run(tf.global_variables_initializer())
n1 = tf.sigmoid(w1*x1 + w2*x2)
print((w1*x1).eval())
print((w2*x2).eval())
print(n1.eval())

import tensorflow as tf

# You can create constants in TF to
# hold specific values
a = tf.constant(1)
b = tf.constant(2)

# Of course you can add, multiply,
# and compute on these as you like
c = a + b
d = a * b

# TF numbers are stored in "tensors",
# a fancy term for multidimensional arrays
# If you pass TF a Python list, it can convert it
V1 = tf.constant([1., 2.])   # Vector, 1-dimensional
V2 = tf.constant([3., 4.])   # Vector, 1-dimensional
M = tf.constant([[1., 2.]])             # Matrix, 2d
N = tf.constant([[1., 2.],[3.,4.]])     # Matrix, 2d
K = tf.constant([[[1., 2.],[3.,4.]]])   # Tensor, 3d+

# You can also compute on tensors
# like you did scalars, but be careful of shape
V3 = V1 + V2

# Operations are element-wise by default
M2 = M * M

# True matrix multiplication requires a special call
NN = tf.matmul(N,N)

# The above code only defines a TF "graph".
# Nothing has been computed yet
# For that, you first need to create a TF "session"
sess = tf.Session()
# Note the parallelism information TF
# reports to you when starting a session

# Now you can run specific nodes of your graph,
# i.e. the variables you've named
output = sess.run(NN)
print("NN is:")
print(output)

# Remember to close your session
# when you're done using it
sess.close()

# Often, we work interactively,
# it's convenient to use a simplified session
sess = tf.InteractiveSession()

# Now we can compute any node
print("M2 is:")
print(M2.eval())

# TF "variables" can change value,
# useful for updating model weights
W = tf.Variable(0, name="weight")

# But variables must be initialized by TF before use
init_op = tf.initialize_all_variables()
sess.run(init_op)

print("W is:")
print(W.eval())

W += a
print("W after adding a:")
print(W.eval())

W += a
print("W after adding a again:")
print(W.eval())

# You can return or supply arbitrary nodes,
# i.e. check an intermediate value or
# sub your value in the middle of a computation

E = d + b # 1*2 + 2 = 4

print("E as defined:")
print(E.eval())

# Let's see what d was at the same time
print("E and d:")
print(sess.run([E,d]))

# Use a custom d by specifying a dictionary
print("E with custom d=4:")
print(sess.run(E, feed_dict = {d:4.}))

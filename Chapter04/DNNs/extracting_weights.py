# Access general TF functions
import tensorflow as tf
import tensorflow.contrib.layers as layers

def conv_learn(X, y, mode):
    # Ensure our images are 2d 
    X = tf.reshape(X, [-1, 36, 36, 1])
    # We'll need these in one-hot format
    y = tf.one_hot(tf.cast(y, tf.int32), 5, 1, 0)

    # conv layer will compute 4 kernels for each 5x5 patch
    with tf.variable_scope('conv_layer'):
        # 5x5 convolution, pad with zeros on edges
        h1 = layers.convolution2d(X, num_outputs=4,
                kernel_size=[5, 5], 
                activation_fn=tf.nn.relu)
        # 2x2 Max pooling, no padding on edges
        p1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID')
                
        # Need to flatten conv output for use in dense layer
    p1_size = np.product(
              [s.value for s in p1.get_shape()[1:]])
    p1f = tf.reshape(p1, [-1, p1_size ])
    
    # densely connected layer with 32 neurons and dropout
    h_fc1 = layers.fully_connected(p1f,
             5,
             activation_fn=tf.nn.relu)
    drop = layers.dropout(h_fc1, keep_prob=0.5, is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)
    
    logits = layers.fully_connected(drop, 5, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(y, logits)
    # Setup the training function manually
    train_op = layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=0.01)
    return tf.argmax(logits, 1), loss, train_op
# Use generic estimator with our function
classifier = estimator.SKCompat(
         learn.Estimator(
         model_fn=conv_learn))

classifier.fit(train,train_labels,
                steps=1024,
                batch_size=32)

# simple accuracy
metrics.accuracy_score(test_labels,classifier.predict(test))

# See layer names
print(classifier._estimator.get_variable_names())




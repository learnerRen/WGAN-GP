import tensorflow as tf
batch_size=64
#input shape [batch_size, 10]

def generator(z_prior, training):
    with tf.variable_scope("genFC_0"):
        w0 = tf.get_variable("w0",
                             [10, 1568],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.variance_scaling_initializer())
        b0 = tf.get_variable("b0",
                             [1568],
                             dtype=tf.float32,
                             initializer=tf.zeros_initializer())
        z0 = tf.matmul(z_prior, w0)
        a0 = tf.nn.leaky_relu(tf.nn.bias_add(z0, b0))
        a0_reshape = tf.reshape(a0, [batch_size, 14, 14, 8])
    with tf.variable_scope("gendeconv_1"):
        w1 = tf.get_variable('w1',
                             [3, 3, 32, 8],#it is opposited to conv2d[h, w ,in, out]  conv2d_tranpose[h, w, out, in]
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.variance_scaling_initializer())
        b1 = tf.get_variable("b1",
                             [32],
                             dtype=tf.float32,
                             initializer=tf.zeros_initializer())
        z1 = tf.nn.conv2d_transpose(a0_reshape,
                                    w1,
                                    [batch_size, 14, 14, 32],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
        a1 = tf.nn.leaky_relu(tf.nn.bias_add(z1,b1))
    with tf.variable_scope("gendeconv_2"):
        w2 = tf.get_variable('w2',
                             [3, 3, 64, 32],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.variance_scaling_initializer())
        b2 = tf.get_variable("b2",
                             [64],
                             dtype=tf.float32,
                             initializer=tf.zeros_initializer())
        z2 = tf.nn.conv2d_transpose(a1,
                                    w2,
                                    [batch_size, 28, 28, 64],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
        a2 = tf.nn.leaky_relu(tf.nn.bias_add(z2, b2))
    """
    with tf.variable_scope('gendeconv_3'):
        w3 = tf.get_variable('w3',
                             [3, 3, 32, 32],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.variance_scaling_initializer())
        b3 = tf.get_variable("b3",
                             [64],
                             dtype=tf.float32,
                             initializer=tf.zeros_initializer())
        z3 = tf.nn.conv2d_transpose(a2,
                                    w3,
                                    [batch_size, 16, 16, 32],#output shape
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
        a3 = tf.nn.leaky_relu(tf.layers.batch_normalization(z3))
    with tf.variable_scope('gendeconv_4'):
        w4 = tf.get_variable('w4',
                             [3, 3, 64, 64],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.variance_scaling_initializer())
        z4 = tf.nn.conv2d_transpose(a2,
                                    w4,
                                    [batch_size, 28, 28, 64],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
        b4 = tf.get_variable("b4",
                             [32],
                             dtype=tf.float32,
                             initializer=tf.zeros_initializer())
        #z4_BN = tf.layers.batch_normalization(z4, training=training)
        a4 = tf.nn.leaky_relu(tf.layers.batch_normalization(z4, training=training))
    """
    with tf.variable_scope('gendeconv_5'):
        w5 = tf.get_variable('w5',
                             [3, 3, 1, 64],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.variance_scaling_initializer())
        b5 = tf.get_variable('b5',
                             [1],
                             dtype=tf.float32,
                             initializer=tf.zeros_initializer())
        z5 = tf.nn.conv2d_transpose(a2,
                                    w5,
                                    [batch_size, 28, 28, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
    return tf.nn.tanh(tf.nn.bias_add(z5,b5))


def discriminator(x, training=True, reuse=False):
    with tf.variable_scope('disconv_1', reuse=reuse):
        d_w1 = tf.get_variable('d_w1',
                               [3, 3, 1, 64],
                               dtype=tf.float32,
                               initializer=tf.contrib.layers.variance_scaling_initializer())
        d_b1 = tf.get_variable("b1",
                               [64],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
        d_z1 = tf.nn.conv2d(x,
                            d_w1,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        d_a1 = tf.nn.leaky_relu(tf.nn.bias_add(d_z1, d_b1))
    with tf.variable_scope('disconv_2', reuse=reuse):
        d_w2 = tf.get_variable('d_w2',
                               [3, 3, 64, 32],
                               dtype=tf.float32,
                               initializer=tf.contrib.layers.variance_scaling_initializer())
        d_b2 = tf.get_variable("b2",
                               [32],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
        d_z2 = tf.nn.conv2d(d_a1,
                            d_w2,
                            strides=[1, 2, 2, 1],
                            padding='SAME')
        d_a2 = tf.nn.leaky_relu(tf.nn.bias_add(d_z2, d_b2))
        """
    with tf.variable_scope('disconv_3', reuse=reuse):
        d_w3 = tf.get_variable('d_w3',
                               [3, 3, 64, 32],
                               dtype=tf.float32,
                               initializer=tf.contrib.layers.variance_scaling_initializer())
        d_b3 = tf.get_variable("d_b3",
                               [32],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
        d_z3 = tf.nn.conv2d(d_a2,
                            d_w3,
                            strides=[1, 2, 2, 1],
                            padding='SAME')
        #d_z3_BN = tf.layers.batch_normalization(d_z3, training=training)
        d_a3 = tf.nn.leaky_relu(tf.layers.batch_normalization(d_z3, training=training))
    """

    with tf.variable_scope('disconv_5', reuse=reuse):
        d_w5 = tf.get_variable('d_w5',
                               [3, 3, 32, 8],
                               dtype=tf.float32,
                               initializer=tf.contrib.layers.variance_scaling_initializer())
        d_z5 = tf.nn.conv2d(d_a2,
                            d_w5,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        d_b5 = tf.get_variable("d_b5",
                               [8],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
        d_a5 = tf.nn.leaky_relu(tf.nn.bias_add(d_z5, d_b5))
        size = d_a5.get_shape()[1]*d_a5.get_shape()[2]*d_a5.get_shape()[3]
        d_a5_flatten = tf.reshape(d_a5, [batch_size, size])
        print(d_a5_flatten)
    with tf.variable_scope('disFC_5', reuse=reuse):
        d_w6 = tf.get_variable('d_w6',
                               [d_a5_flatten.get_shape()[1], 1],
                               dtype=tf.float32,
                               initializer=tf.contrib.layers.variance_scaling_initializer())
        d_b6 = tf.get_variable('d_b4',
                               [1],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
        d_z6 = tf.matmul(d_a5_flatten, d_w6)+d_b6
    return d_z6
















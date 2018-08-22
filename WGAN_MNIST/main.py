from model import *
import os
import numpy as np
import time
from keras.datasets import mnist
from utils import *
to_restore = False
weights_output_path = 'weights_output'
model_name = 'my_GAN'
image_output = 'image_out'
batch_size = 64
max_iteration = 1000000
(train_image, train_label), (test_image, test_label) = mnist.load_data()
image = np.concatenate([train_image[:, :, :, np.newaxis], test_image[:, :, :, np.newaxis]], axis=0).astype(np.float32)/127.5-1
#print(image.shape)
image_num = 70000
#noise = np.random.rand(image_num, 28, 28, 1).astype(np.float32)*0.2-0.1
#image += noise
reuse = True
times = 5 #update discriminator times times, update generator once
lambd = 10
if os.path.isdir(image_output):
    pass
else:
    os.mkdir(image_output)
z_prior = tf.placeholder(dtype=tf.float32,
                         shape=[batch_size, 10])
x = tf.placeholder(dtype=tf.float32,
                   shape=[batch_size, 28, 28, 1])
training = tf.placeholder(tf.bool)
global_steps = tf.Variable(tf.constant(-1000, dtype=tf.int32), trainable=False)
x_generate = generator(z_prior, training=training)
print(x_generate)
y_real = discriminator(x, training=training)
y_generate = discriminator(x_generate, training=training, reuse=True)
# the loss of WGAN-GP is the same like WGAN
d_loss = tf.reduce_mean(y_generate)-tf.reduce_mean(y_real)
g_loss = -tf.reduce_mean(y_generate)
differences = x_generate - x
alpha = tf.random_uniform([batch_size, 1, 1, 1], minval=0., maxval=1.)
interpolates = x + alpha*differences
grad = tf.gradients(discriminator(interpolates, training=training, reuse=True), [interpolates])[0]
grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), reduction_indices=[1, 2, 3]))
grad_penality = tf.reduce_mean((grad_norm-1.)**2)
d_loss += lambd * grad_penality
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'dis' in var.name]
g_vars = [var for var in t_vars if 'gen' in var.name]
optimizer = tf.train.AdamOptimizer(0.0001, 0.5, 0.9)
#with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
d_train = optimizer.minimize(d_loss, var_list=d_vars)
g_train = optimizer.minimize(g_loss, var_list=g_vars)
start = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if to_restore:
        check_file = tf.train.latest_checkpoint(weights_output_path)
        saver.restore(sess, check_file)
    else:
        if os.path.exists(weights_output_path):
            pass
        else:
            os.mkdir(weights_output_path)
    z_sample = np.random.normal(0, 1, [batch_size, 10]).astype(np.float32)
    for i in range(max(0, sess.run(global_steps)), max_iteration):
        a = np.random.randint(0, image_num, [batch_size], dtype=np.int)
        z_value = np.random.uniform(-1, 1, size=[batch_size, 10]).astype(np.float32)
        sess.run(d_train, feed_dict={z_prior: z_value, x: (image[a, :, :, :]), training: True})
        if i % times == 0:
            a = np.random.randint(0, image_num, [batch_size], dtype=np.int)
            z_value = np.random.uniform(-1, 1, size=[batch_size, 10]).astype(np.float32)
            sess.run(g_train, feed_dict={z_prior: z_value, x: (image[a, :, :, :]), training: True})
        if i % 1000 == 0:
            #image -= noise
            #noise = np.random.rand(image_num, 28, 28, 1).astype(np.float32) * 0.2 - 0.1
            #image += noise
            sess.run(tf.assign(global_steps, sess.run(global_steps) + 1000))
            print("training {} times".format(sess.run(global_steps)))
        if i % 10000 == 0:
            end = time.time()
            d_loss_value = sess.run(d_loss, feed_dict={z_prior: z_value, x: image[a, :, :, :], training: False})
            g_loss_value = sess.run(g_loss, feed_dict={z_prior: z_value, x: image[a, :, :, :], training: False})
            print("After {} training times, the d_loss is {}, and the g_loss is {} using {} seconds".format(i, d_loss_value, g_loss_value, end-start))
            start = end
            x_generate_value = sess.run(x_generate, feed_dict={z_prior: z_value, training: False})
            x_generate_value = x_generate_value[0, :, :, :]
            x_generate_value = np.squeeze(x_generate_value)
            print(x_generate_value)
            global_steps_value = sess.run(global_steps)
            saver.save(sess, os.path.join(weights_output_path, model_name), global_steps_value)
            show_result(x_generate_value, os.path.join(image_output, "random_image",
                                                       "ramdom"+str(global_steps_value)+'.jpg'))
            print("random image{} is saved ".format(int(global_steps_value/10000)))
            x_sample = sess.run(x_generate, feed_dict={z_prior: z_sample, training: False})
            x_sample = x_sample[0, :, :, :]
            x_sample = np.squeeze(x_sample)
            show_result(x_sample, os.path.join(image_output, "sample_image", "sample"+str(global_steps_value)+'.jpg'))
            print("sample image{} is saved ".format(int(global_steps_value / 10000)))

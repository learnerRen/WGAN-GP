# WGAN-GP
## take care about the loss
d_loss = y_generate - y_real
g_loss = -y_generate
## penalty
pen = gradient of interpolates(x - randondom_uniform*difference(x-x_generate))
d_loss += 10*(sqrt(tf.reduce_sum(square(pen)-1, reduction_indices=[range(1, generate_dim)])**2
## update times
update discriminator 5 times, update generator once
## optimizer
Although the author of WGAN advice us to use RMS rather than other optimizer method which use moment, I find it seems that adam(I change the default hyperparameter, and set learning_rate=1e-4, alpha=0.5, beta=0.9) coverge quicker. It does not matter if you try RMSprop. Attention: when we minimize g_loss we will not update the parameters in discriminator. Similarly, when we minimize d_loss we will not update the parameters in generator.
## noise
For MNIST and cifar, the number of samples is large, so the noise seems to useless, but for my animal data set(dog and cat almost 15 thousand samples), noise will be helpful.
## about the parameter training
At first, I intend to use the as the parameters of batch_normalization, I do not delete it in order to use is in the future.
## Discriminator and generator:
deconv is an important part of generaor
The discriminator and the generator is highly symmetric. However, it is also fine that discriminator is a little stronger than generator.
## training times
I use a GTX1080Ti, and the generators and the disriminators was different in different data sets
For MNIST, every 10 thousand iteration is about 120s
For cifar10, every 10 thousand iteration is about 170s
# results 



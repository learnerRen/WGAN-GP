# WGAN-GP
## Prerequisites
### python 3.x 
### tensorflow 1.7 
## take care about the loss
d_loss = y_generate - y_real <br>
g_loss = -y_generate <br>
## penalty
pen = gradient of interpolates(x - randondom_uniform * difference(x-x_generate))<br />
d_loss += 10*(sqrt(tf.reduce_sum(square(pen)-1, reduction_indices=[range(1, generate_dim)])**2
## update times
update discriminator 5 times, update generator once
## optimizer
Although the author of WGAN advice us to use RMS rather than other optimizer method which use moment, I find it seems that adam(I change the default hyperparameter, and set learning_rate=1e-4, alpha=0.5, beta=0.9) coverge quicker.\<br>
It does not matter if you try RMSprop. Attention: when we minimize g_loss we will not update the parameters in discriminator. Similarly, when we minimize d_loss we will not update the parameters in generator.
## noise
For MNIST and cifar, the number of samples is large, so the noise seems to useless, but for my animal data set(dog and cat almost 15 thousand samples), noise will be helpful.
## about the parameter training
At first, I intend to use the as the parameters of batch_normalization, I do not delete it in order to use is in the future.
## Discriminator and generator:
deconv is an important part of generaor
The discriminator and the generator is highly symmetric. However, it is also fine that discriminator is a little stronger than generator.
## training times
I use a GTX1080Ti, and the generators and the disriminators was different in different data sets.\<br>
For MNIST, the time of every 10 thousand iteration is about 120s\<br>
For cifar10, the time of every 10 thousand iteration is about 170s\<br>
For animal data set, the time of every 10 thousand iteration is about 2700s
# results 
![image](https://github.com/learnerRen/WGAN-GP/blob/master/WGAN_MNIST/image_out/random_image/ramdom0.jpg)
![image](https://github.com/learnerRen/WGAN-GP/blob/master/WGAN_MNIST/image_out/random_image/ramdom50000.jpg)
![image](https://github.com/learnerRen/WGAN-GP/blob/master/WGAN_MNIST/image_out/random_image/ramdom110000.jpg)


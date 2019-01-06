# coding:utf-8

import matplotlib, os
matplotlib.use("Agg")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from utils import * 


class GAN(BasicTrainFramework):
    def __init__(self, 
                gan_type='gan',
                optim_type='adam',
                data=datamanager_gaussian(0,1),
                batch_size=64, 
                noise_dim=50,
                learning_rate=2e-4,
                optim_num=0.5,
                clip_num=0.03,
                critic_iter=5
                ):
        
        self.noise_dim = noise_dim
        self.clip_num = None if clip_num==0 else clip_num
        self.lr = learning_rate
        self.optim_num = optim_num
        self.critic_iter = critic_iter
        super(GAN, self).__init__(batch_size, gan_type)
        
        self.gan_type = gan_type
        self.optim_type = optim_type

        self.data = data

        np.random.seed(233)
        # self.sample_data = np.random.uniform(-1.0, 1.0, (self.batch_size, self.noise_dim))
        self.sample_data = np.random.normal(size=(self.batch_size, self.noise_dim))

        self.generator = Generator_MLP(name='generator')
        self.discriminator = Discriminator_MLP(name='discriminator')

        self.build_placeholder()
        self.build_gan()
        self.build_optimizer(optim_type)
        self.build_summary()

        self.build_sess()
        self.build_dirs()
    
    def build_placeholder(self):
        self.noise = tf.placeholder(shape=(self.batch_size, self.noise_dim), dtype=tf.float32)
        self.source = tf.placeholder(shape=(self.batch_size, self.noise_dim), dtype=tf.float32)

    def build_gan(self):
        self.G = self.generator(self.noise, is_training=True, reuse=False)
        self.G_test = self.generator(self.noise, is_training=False, reuse=True)
        self.logit_real, self.net_real = self.discriminator(self.source, is_training=True, reuse=False)
        self.logit_fake, self.net_fake = self.discriminator(self.G, is_training=True, reuse=True)

        self.mean_real, self.std_real = tf.nn.moments(self.source, axes=[0,1])
        self.mean_fake, self.std_fake = tf.nn.moments(self.G, axes=[0,1])
        self.std_real = tf.sqrt(self.std_real)
        self.std_fake = tf.sqrt(self.std_fake)

    def build_optimizer(self, optim_type='adam'):
        if self.gan_type == 'gan':
            self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_real, labels=tf.ones_like(self.logit_real)))
            self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.zeros_like(self.logit_fake)))
            self.D_loss = self.D_loss_real + self.D_loss_fake
            self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.ones_like(self.logit_fake)))
        elif self.gan_type == 'wgan':
            self.D_loss_real = - tf.reduce_mean(self.logit_real) 
            self.D_loss_fake = tf.reduce_mean(self.logit_fake)
            self.D_loss = self.D_loss_real + self.D_loss_fake
            self.G_loss = - self.D_loss_fake
            if self.clip_num:
                print "GC"
                self.D_clip = [v.assign(tf.clip_by_value(v, -self.clip_num, self.clip_num)) for v in self.discriminator.weights]
        elif self.gan_type == 'lsgan':
            def mse_loss(pred, data):
                return tf.sqrt(2 * tf.nn.l2_loss(pred - data)) / self.batch_size
            self.D_loss_real = tf.reduce_mean(mse_loss(self.logit_real, tf.ones_like(self.logit_real)))
            self.D_loss_fake = tf.reduce_mean(mse_loss(self.logit_fake, tf.zeros_like(self.logit_fake)))
            self.D_loss = 0.5 * (self.D_loss_real + self.D_loss_fake)
            self.G_loss = tf.reduce_mean(mse_loss(self.logit_fake, tf.ones_like(self.logit_fake)))
            if self.clip_num:
                print "GC"
                self.D_clip = [v.assign(tf.clip_by_value(v, -self.clip_num, self.clip_num)) for v in self.discriminator.weights]

        if optim_type == 'adam':
            self.D_solver = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.optim_num).minimize(self.D_loss, var_list=self.discriminator.vars)
            self.G_solver = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.optim_num).minimize(self.G_loss, var_list=self.generator.vars)
        elif optim_type == 'rmsprop':
            self.D_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.discriminator.vars)
            self.G_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list=self.generator.vars)

    def build_summary(self):
        D_sum = tf.summary.scalar("D_loss", self.D_loss)
        D_sum_real = tf.summary.scalar("D_loss_real", self.D_loss_real)
        D_sum_fake = tf.summary.scalar("D_loss_fake", self.D_loss_fake)
        G_sum = tf.summary.scalar("G_loss", self.G_loss)
        mean_sum_real = tf.summary.scalar("mean_real", self.mean_real)
        mean_sum_fake = tf.summary.scalar("mean_fake", self.mean_fake)
        std_sum_real = tf.summary.scalar("std_real", self.std_real)
        std_sum_fake = tf.summary.scalar("std_fake", self.std_fake)
        self.summary = tf.summary.merge([D_sum, D_sum_real, D_sum_fake, G_sum, 
                                        mean_sum_real, mean_sum_fake, std_sum_real, std_sum_fake])
        
    def test(self):
        out = self.sess.run(self.G_test, feed_dict={self.noise:self.sample_data})
        print "mean=%.2f std=%.2f" % (np.mean(out), np.std(out))
    
    def sample(self, epoch):
        real = self.data([500000, self.noise_dim])
        pr, _ = np.histogram(real, bins=np.linspace(-6, 10, 200), density=True)
        plt.plot(np.linspace(-6, 10, len(pr)), pr, label='real', color='g', linewidth=2)
        fake = []
        for i in range(500):
            out = self.sess.run(self.G_test, feed_dict={self.noise: np.random.normal(size=(self.batch_size, self.noise_dim))})
            fake.append(out)
        pf, _ = np.histogram(np.concatenate(fake), bins=np.linspace(-6, 10, 200), density=True)
        plt.plot(np.linspace(-6, 10, len(pf)), pf, label='fake', color='r', linewidth=1.5)
        plt.title("epoch_{}".format(epoch))
        plt.legend()
        plt.savefig(os.path.join(self.fig_dir, "epoch_{}.png".format(epoch)))
        plt.clf()

    def train(self, epoches=1):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        batches_per_epoch = 100

        for epoch in range(epoches):

            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx 

                data = self.data([self.batch_size, self.noise_dim])
                
                feed_dict = {
                    self.source: data,
                    self.noise: np.random.normal(size=(self.batch_size, self.noise_dim))
                }
                
                # train D
                self.sess.run(self.D_solver, feed_dict=feed_dict)
                if self.clip_num:
                    self.sess.run(self.D_clip)

                # train G
                if (cnt-1) % self.critic_iter == 0:
                    self.sess.run(self.G_solver, feed_dict=feed_dict)
                    
                if cnt % 20 == 0:
                    d_loss, d_loss_r, d_loss_f, g_loss, sum_str = self.sess.run([self.D_loss, self.D_loss_real, self.D_loss_fake, self.G_loss, self.summary], feed_dict=feed_dict)
                    print self.version + " epoch [%3d/%3d] iter [%3d/%3d] D=%.3f Dr=%.3f Df=%.3f G=%.3f" % \
                        (epoch, epoches, idx, batches_per_epoch, d_loss, d_loss_r, d_loss_f, g_loss)
                    self.writer.add_summary(sum_str, cnt)
            self.test()
            if epoch % 25 == 0:
                self.sample(epoch)
        self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)

if __name__ == "__main__":

    data = datamanager_gaussian(mean=3.5, std=0.7)

    gan = GAN(gan_type='gan', data=data, batch_size=64, noise_dim=10, clip_num=0, critic_iter=5)
    gan.train(100)
    
    gan = GAN(gan_type='wgan', data=data, batch_size=64, noise_dim=10, clip_num=0.1, critic_iter=5)
    gan.train(100)

    gan = GAN(gan_type='lsgan', data=data, batch_size=64, noise_dim=10, clip_num=0.1, critic_iter=5)
    gan.train(100)
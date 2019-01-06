# coding:utf-8

import tensorflow as tf 
import numpy as np
import os

class BasicBlock(object):
    def __init__(self, hidden_units, name):
        self.name = name
        self.hidden_units = hidden_units
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

class BasicTrainFramework(object):
	def __init__(self, batch_size, version):
		self.batch_size = batch_size
		self.version = version

	def build_dirs(self):
		self.log_dir = os.path.join('logs', self.version) 
		self.model_dir = os.path.join('checkpoints', self.version)
		self.fig_dir = os.path.join('figs', self.version)
		for d in [self.log_dir, self.model_dir, self.fig_dir]:
			if (d is not None) and (not os.path.exists(d)):
				print "mkdir " + d
				os.makedirs(d)
	
	def build_sess(self):
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
	
	def build_network(self):
		self.D_logit_real = None 
		self.D_logit_fake = None

	def load_model(self, checkpoint_dir=None, ckpt_name=None):
		import re 
		print "load checkpoints ..."
		checkpoint_dir = checkpoint_dir or self.model_dir
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = ckpt_name or os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print "Success to read {}".format(ckpt_name)
			return True, counter
		else:
			print "Failed to find a checkpoint"
			return False, 0

def bn(x, is_training, name):
	return tf.contrib.layers.batch_norm(x, 
										decay=0.999, 
										updates_collections=None, 
										epsilon=0.001, 
										scale=True,
										fused=False,
										is_training=is_training,
										scope=name)

def spectral_norm(w, iteration=10, name="sn"):
	'''
	Ref: https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/65218e8cc6916d24b49504c337981548685e1be1/spectral_norm.py
	'''
	w_shape = w.shape.as_list() # [KH, KW, Cin, Cout] or [H, W]
	w = tf.reshape(w, [-1, w_shape[-1]]) # [KH*KW*Cin, Cout] or [H, W]

	u = tf.get_variable(name+"_u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
	s = tf.get_variable(name+"_sigma", [1, ], initializer=tf.random_normal_initializer(), trainable=False)

	u_hat = u # [1, Cout] or [1, W]
	v_hat = None 

	for _ in range(iteration):
		v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(w))) # [1, KH*KW*Cin] or [1, H]
		u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w)) # [1, Cout] or [1, W]
		
	u_hat = tf.stop_gradient(u_hat)
	v_hat = tf.stop_gradient(v_hat)

	sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat)) # [1,1]
	sigma = tf.reshape(sigma, (1,))

	with tf.control_dependencies([u.assign(u_hat), s.assign(sigma)]):
		# ops here run after u.assign(u_hat)
		w_norm = w / sigma 
		w_norm = tf.reshape(w_norm, w_shape)
	
	return w_norm

def dense(x, output_size, stddev=0.02, bias_start=0.0, activation=None, sn=False, reuse=False, name='dense'):
	shape = x.get_shape().as_list()
	with tf.variable_scope(name, reuse=reuse):
		W = tf.get_variable(
			'weights', [shape[1], output_size], 
			tf.float32, 
			tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable(
			'biases', [output_size], 
			initializer=tf.constant_initializer(bias_start))
		if sn:
			W = spectral_norm(W, name="sn")
	out = tf.matmul(x, W) + bias 
	if activation is not None:
		out = activation(out)
	
	return out

def conv_cond_concat(x, y):
    # x: [N, H, W, C]
    # y: [N, 1, 1, d]
	x_shapes = x.get_shape()
	y_shapes = y.get_shape()
	return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

class Generator_MLP(BasicBlock):
    def __init__(self, name=None):
        super(Generator_MLP, self).__init__(None, name or "Generator_MLP")
    
    def __call__(self, z, y=None, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            if y is not None:
                z = tf.concat([z,y], 1)
            net = tf.nn.softplus(dense(z, 64, name='g_fc1'))
            out = dense(net, 10, name='g_fc2')
            return out

class Discriminator_MLP(BasicBlock):
    def __init__(self, class_num=None, name=None):
        super(Discriminator_MLP, self).__init__(None, name or "Discriminator_MLP")
        self.class_num = class_num
    
    def __call__(self, x, y=None, sn=False, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            batch_size = x.get_shape().as_list()[0]
            if y is not None:
                ydim = y.get_shape().as_list()[-1]
                y = tf.reshape(y, [batch_size, 1, 1, ydim])
                x = conv_cond_concat(x, y) # [bz, 28, 28, 11]

            net = tf.nn.tanh(dense(x, 64, sn=sn, name='d_fc1'))
            net = tf.nn.tanh(bn(dense(net, 64, sn=sn, name='d_fc2'), is_training, name='d_bn2'))

            yd = dense(net, 1, sn=sn, name="D_dense")
            
            if self.class_num is not None:
                print self.class_num
                yc = dense(net, self.class_num, sn=sn, name='C_dense')
                return yd, net, yc 
            else:
                return yd, net
    
    @property
    def weights(self):
        res = []
        for v in self.vars:
            if "weights" in v.name:
                res.append(v)
        return res

class datamanager_gaussian(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std 
    
    def __call__(self, size):
        return np.random.normal(self.mean, self.std, size=size)
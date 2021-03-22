'''
Network
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import logging
from math import ceil
import sys

import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

class FCN8VGG:
	"""docstring for ClassName"""
	def __init__(self,vgg16_npy_path="vgg16.npy"):
		path = "weights"
		path = os.path.join(path,vgg16_npy_path)
		logging.info("VGG16 load Np file:",path)
		if not os.path.isfile(path):
			logging.error("File Not Found")
		self.data_dict = np.load(path,encoding='latin1').item()
		self.wd = 5e-4
		logging.info("NPY file loaded")
		pass


	def weight_variable(self,shape):
		print ("$$$$$$$$$$$$$$$$$shape:,",shape)
		initer =tf.contrib.layers.xavier_initializer(
					uniform=True,
					seed=None,
					# shape=shape,
					dtype=tf.float32
				)
		return tf.get_variable("W",
							   dtype=tf.float32,
							   shape=shape,
							   initializer=initer)

	def bias_variable(self,shape):
		initial = tf.constant(0.,shape=shape,dtype=tf.float32)
		return tf.get_variable('b',
							   dtype=tf.float32,
							   initializer=initial)

	def max_pool(self,x,name,debug=False):
		ksize_=2
		stride=2
		pool=tf.nn.max_pool(x,
							  ksize=[1,ksize_,ksize_,1],
							  strides=[1,stride,stride,1],
							  padding="SAME",
							  name=name)

		if debug:
			pool=tf.Print(pool,[tf.shape(pool)],
							message='shape of %s'%name,
							summarize=4,first_n=1)
		return pool

	def get_fc_weight_reshape(self,name,shape):
		print ('Layer name: %s'%name)
		print ('Layer shape: %s'%shape)

		weights = self.data_dict[name][0]
		weights = weights.reshape(shape)

		init = tf.constant_initializer(value=weights,dtype=tf.float32)
		var = tf.get_variable(name='weights',initializer=init,shape=shape)

		return var




	def fc_layer(self,x,name,num_classes=None,use_relu=True):
		with tf.variable_scope(name) as scope:
			shape = x.get_shape().as_list()

			if name == 'fc6':
				filt = self.get_fc_weight_reshape(name,[7,7,512,4096])
			elif name == 'score_fr':
				name = 'fc8'
				filt = self.get_fc_weight_reshape(name,[1,1,4096,1000],num_classes=num_classes)
			else:
				filt = self.get_fc_weight_reshape(name,[1,1,4096,4096])

			# self.__add_wd_and_summary(filt,self.wd,"fc_losses")

			conv = tf.nn.conv2d(x,filt,[1,1,1,1],padding='SAME')
			conv_biases=self.get_bias(name,num_classes=num_classes)
			bias = tf.nn.bias_add(conv,conv_biases)

			if use_relu:
				bias = tf.nn.relu(bias)


			return bias

			# __activation_summary(bias)

			# in_dim=x.get_shape()[1]
			# W=self.weight_variable(shape=[in_dim,num_units])
			# tf.summary.histogram('weights',W)
			# b=self.bias_variable(shape=[num_units])
			# tf.summary.histogram('bias',b)
			# layer = tf.matmul(x,W)
			# layer +=b
			# if use_relu:
			# 		layer = tf.nn.relu(layer)


			

	def get_bias(self,name,num_classes=2):
		bias_weights = self.data_dict[name][1]
		shape = self.data_dict[name][1].shape
		if name == 'fc8':

			bias_weights = self.bias_reshape(bias_weights,shape[0],num_classes)
			shape = [num_classes]
		init = tf.constant_initializer(value=bias_weights,
										dtype=tf.float32)
		var = tf.get_variable(name='biases',initializer=init,shape=shape)
		# __variable_summary(var)
		return var


	def get_conv_filter(self,name):
		init = tf.constant_initializer(value = self.data_dict[name][0], dtype= tf.float32)
		shape = self.data_dict[name][0].shape
		print('Layer name:%s'%name)
		print('Weights/Filter shape: ',shape)
		var = tf.get_variable(name='filter',initializer=init,shape=shape)
		if not tf.get_variable_scope().reuse:
			weight_Decay = tf.multiply(tf.nn.l2_loss(var),self.wd,
				name='weight_loss')
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weight_Decay)
		# __variable_summary(var)
		return var

	def conv_layer(self,x,name,load_weights):
		with tf.variable_scope(name):
			if load_weights:
				with tf.variable_scope(name) as scope:
					filt = self.get_conv_filter(name)
					print ("CONV Filt:",filt)
					conv = tf.nn.conv2d(x,filt,[1,1,1,1],padding='SAME')
					conv_biases = self.get_bias(name)
					bias = tf.nn.bias_add(conv,conv_biases)
					relu = tf.nn.relu(bias)
					return relu
					# __activation_summary(relu)
			# num_in_channel=x.get_shape().as_list()[-1] ################################################# check
			# shape = [filter_size,filter_size,num_in_channel,num_filters]
			# W=self.weight_variable(shape=shape)
			# tf.summary.histogram("weight",W)
			# b=self.bias_variable(shape=[num_filters])
			# tf.summary.histogram("bias",b)
			# layer = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding="SAME")
			# layer +=b
			# return tf.nn.tanh(layer,name="tanh_out")


	def variable_with_weight_decay(self,shape,stddev,wd,decoder=False):
		initializer = tf.truncated_normal_initializer(stddev=stddev)
		var = tf.get_variable('weights',shape=shape,initializer=initializer)
		collection_name = tf.GraphKeys.REGULARIZATION_LOSSES

		if wd and (not tf.get_variable_scope().reuse):
			weight_decay= tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
			tf.add_to_collection(collection_name,weight_decay)

		# __variable_summary(var)
		return var

	def _bias_variable(self,shape,constant=0.0):
		initializer = tf.constant_initializer(constant)
		var = tf.get_variable(name='biases',shape=shape,
								initializer=initializer)
		# __variable_summary(var)
		return var

	def score_layer(self,x,name,num_classes):	
		with tf.variable_scope(name) as scope:
			in_features = x.get_shape()[3].value

			shape = [1,1,in_features,num_classes]
			if name == 'score_fr':
				num_input = in_features
				stddev = (2/num_input)**0.5
			if name == 'score_pool4':
				stddev = 0.001
			if name == "score_pool3":
				stddev = 0.0001
			w_decay = self.wd
			weights = self.variable_with_weight_decay(shape,stddev,w_decay,decoder=True)
			conv = tf.nn.conv2d(x,weights,[1,1,1,1],padding='SAME')
			conv_biases = self._bias_variable([num_classes],constant=0.0)
			bias = tf.nn.bias_add(conv,conv_biases)
			return bias
			# _acitivation_summary(bias)

	def get_deconv_filter(self,f_shape):
		width = f_shape[0]
		height = f_shape[0]
		f=ceil(width/2.0)
		c = (2*f - 1 -f %2)/(2.0*f)
		bilinear = np.zeros([f_shape[0],f_shape[1]])
		for x in range(width):
			for y in range(height):
				value = (1 - abs( x / f -c))*(1-abs(y / f - c))
				bilinear[x,y]=value
		weights = np.zeros(f_shape)
		for i in range(f_shape[2]):
			weights[:,:,i,i] = bilinear 
		init = tf.constant_initializer(value=weights,dtype=tf.float32)
		var = tf.get_variable(name='up_filter',initializer=init,shape=weights.shape)
		return var


	def upscore_layer(self,x,shape,num_classes,name,debug,ksize=4,stride=2):
		strides = [1,stride,stride,1]
		with tf.variable_scope(name):
			in_features = x.get_shape()[3].value
			if shape is None:
				in_shape=tf.shape(x)
				h = ((in_shape[1]-1)*stride) + 1
				w = ((in_shape[2]-1)*stride) + 1
				new_shape = [in_shape[0],h,w,num_classes]
			else:
				new_shape = [shape[0],shape[1],shape[2],num_classes]
			output_shape = tf.stack(new_shape)
			logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
			f_shape = [ksize,ksize,num_classes,in_features]
			num_input = ksize*ksize*in_features/stride
			stddev = (2/num_input)**0.5
			weights = self.get_deconv_filter(f_shape)
			# self.__add_wd_and_summary(weights,self.wd,"fc_classes")
			deconv = tf.nn.conv2d_transpose(x,weights,output_shape,strides,padding='SAME')
			if debug:
				deconv == tf.Print(deconv,[tf.shape(deconv)],
									message='shape of %s' % name,
									summarize=4,first_n=1)
		# _activation_summary(deconv)
		return deconv


	def add_softmax(self,num_classes, logits):
		num_classes = num_classes
		with tf.name_scope('decoder'):
			logits = tf.reshape(logits, (-1, num_classes))
			epsilon = tf.constant(value=0.000000001)
			# logits = logits + epsilon

			softmax = tf.nn.softmax(logits)

		return softmax



	def dense_layer(self,inp_layer,out_nodes,name,use_relu=True,use_leaky=False):
		with tf.variable_scope(name):
			in_dim=inp_layer.get_shape()[1]
			W=self.weight_variable(shape=[in_dim,out_nodes])
			tf.summary.histogram('weights',W)
			b=self._bias_variable(shape=[out_nodes],constant=0.0)
			tf.summary.histogram('bias',b)
			layer = tf.matmul(inp_layer,W)
			layer +=b
			if use_relu:
				layer = tf.nn.elu(layer)
			if use_leaky:
				layer=tf.nn.leaky_relu(layer)		
										
			return layer

	def build_decoder(self,num_classes):
		
		logits={}
		fc_layer = 'pool5'
		if fc_layer == 'pool5':
			logits['fcn_in'] = self.pool5
		elif fc_layer == 'fc7':
			logits['fcn_in'] = self.fc7
		else: 
			raise NotImplementedError

		logits['feed2'] = self.pool4
		logits['feed4'] = self.pool3
		logits['fcn_logits'] = self.upscore32
		logits['flatten']=self.flatten_layer_branch
		fcn_in = logits['fcn_in']
		


		sd=0.01
		with tf.variable_scope("Decoder") as scope:
			he_init = tf.contrib.layers.variance_scaling_initializer()

			l2_regularizer = tf.contrib.layers.l2_regularizer(5e-4)

			# Build score_fr layer
			score_fr = tf.layers.conv2d(
				fcn_in, kernel_size=[1, 1], filters=num_classes, padding='SAME',
				name='score_fr', kernel_initializer=he_init,
				kernel_regularizer=l2_regularizer)

			# _activation_summary(score_fr)

			# Do first upsampling

			upscore2 = self.upscore_layer(score_fr, 
											shape=tf.shape(logits['feed2']),
											num_classes=num_classes,
											name='upscore2', 
											debug=False,
											ksize=4, 
											stride=2)
			print ("Upscore2",upscore2)
			he_init2 = tf.contrib.layers.variance_scaling_initializer(factor=2.0*sd)
			# Score feed2
			score_feed2 = tf.layers.conv2d(
				logits['feed2'], kernel_size=[1, 1], filters=num_classes,
				padding='SAME', name='score_feed2', kernel_initializer=he_init2,
				kernel_regularizer=l2_regularizer)

			# _activation_summary(score_feed2)
			skip = True
			if skip:
				# Create skip connection
				fuse_feed2 = tf.add(upscore2, score_feed2)
			else:
				fuse_feed2 = upscore2
				fuse_feed2.set_shape(score_feed2.shape)

			# Do second upsampling
			upscore4 = self.upscore_layer(fuse_feed2, 
											shape=tf.shape(logits['feed4']),
											num_classes=num_classes, 
											name='upscore4', 
											debug=False,
											ksize=4, 
											stride=2)
			print ("Upscore4",upscore4)
			he_init4 = tf.contrib.layers.variance_scaling_initializer(factor=2.0*sd*sd)
			# Score feed4
			score_feed4 = tf.layers.conv2d(
				logits['feed4'], kernel_size=[1, 1], filters=num_classes,
				padding='SAME', name='score_feed4', kernel_initializer=he_init4,
				kernel_regularizer=l2_regularizer)

			# _activation_summary(score_feed4)

			if skip:
				# Create second skip connection
				fuse_pool3 = tf.add(upscore4, score_feed4)
			else:
				fuse_pool3 = upscore4
				fuse_pool3.set_shape(score_feed4.shape)

			# Do final upsampling
			upscore32 = self.upscore_layer(fuse_pool3, 
											shape=tf.shape(self.bgr),
											num_classes=num_classes, 
											name='upscore32', 
											debug=False,
											ksize=16, 
											stride=8)
			print ("Upscore32",upscore32)



			fc1 = self.dense_layer(logits['flatten'],1024*5,"Flattern_FC1",use_relu=True)
			fc2 = self.dense_layer(fc1,1024*2,"Flatten_FC2",use_relu=True)
			fc2_1 = self.dense_layer(fc2,1024,"Flatten_FC2_1",use_relu=True)
			fc3_1 = self.dense_layer(fc2_1,512,"Flatten_FC3_1",use_relu=True)
			fc3 = self.dense_layer(fc3_1,256,"Flatten_FC3",use_relu=True)
			fc4_1 = self.dense_layer(fc3,128,"Flatten_FC4_1",use_relu=True)
			fc4_2 = self.dense_layer(fc3,128,"Flatten_FC4_2",use_relu=True)
			fc4_3 = self.dense_layer(fc3,128,"Flatten_FC4_3",use_relu=True)
			fc4_4 = self.dense_layer(fc3,128,"Flatten_FC4_4",use_relu=True)

			fc4_1 = self.dense_layer(fc4_1,64,"Flatten_FC6_1",use_relu=True)
			fc4_2 = self.dense_layer(fc4_2,64,"Flatten_FC6_2",use_relu=True)
			fc4_3 = self.dense_layer(fc4_3,64,"Flatten_FC6_3",use_relu=True)
			fc4_4 = self.dense_layer(fc4_4,64,"Flatten_FC6_4",use_relu=True)


			fc5_1 = self.dense_layer(fc4_1,32,"Flatten_FC5_1",use_relu=False,use_leaky=True)
			fc5_2 = self.dense_layer(fc4_2,32,"Flatten_FC5_2",use_relu=False,use_leaky=True)
			fc5_3 = self.dense_layer(fc4_3,32,"Flatten_FC5_3",use_relu=False,use_leaky=True)
			fc5_4 = self.dense_layer(fc4_4,32,"Flatten_FC5_4",use_relu=False,use_leaky=True)


			decoded_logits = {}
			decoded_logits['logits'] = upscore32
			decoded_logits['fc_5_1'] = fc5_1
			decoded_logits['fc_5_2'] = fc5_2
			decoded_logits['fc_5_3'] = fc5_3
			decoded_logits['fc_5_4'] = fc5_4
			decoded_logits['softmax'] = self.add_softmax(num_classes, upscore32)




		return logits,decoded_logits



	def build_model(self, rgb, train=False, num_classes=20, random_init_fc8=False,
			  debug=False):
		"""
		Build the VGG model using loaded weights
		Parameters
		----------
		rgb: image batch tensor
			Image in rgb shap. Scaled to Intervall [0, 255]
		train: bool
			Whether to build train or inference graph
		num_classes: int
			How many classes should be predicted (by fc8)
		random_init_fc8 : bool
			Whether to initialize fc8 layer randomly.
			Finetuning is required in this case.
		debug: bool
			Whether to print additional Debug Information.
		"""
		with tf.name_scope("Pre_Processing"):
			red,green,blue = tf.split(rgb,3,3)
			self.bgr = tf.concat([
							blue - VGG_MEAN[0],
							green - VGG_MEAN[0],
							red - VGG_MEAN[0]],3)
		self.conv1_1 = self.conv_layer(self.bgr,name="conv1_1",load_weights = True)
		self.conv1_2 = self.conv_layer(self.conv1_1,name="conv1_2",load_weights = True)
		self.pool1 = self.max_pool(self.conv1_2,name = "pool1")

		self.conv2_1 = self.conv_layer(self.pool1,name="conv2_1",load_weights = True)
		self.conv2_2 = self.conv_layer(self.conv2_1,name="conv2_2",load_weights = True)
		self.pool2 = self.max_pool(self.conv2_2,name = "pool2")

		self.conv3_1 = self.conv_layer(self.pool2,name="conv3_1",load_weights = True)
		self.conv3_2 = self.conv_layer(self.conv3_1,name="conv3_2",load_weights = True)
		self.conv3_3 = self.conv_layer(self.conv3_2,name="conv3_3",load_weights = True)
		self.pool3 = self.max_pool(self.conv3_3,name = "pool3")

		self.conv4_1 = self.conv_layer(self.pool3,name="conv4_1",load_weights = True)
		self.conv4_2 = self.conv_layer(self.conv4_1,name="conv4_2",load_weights = True)
		self.conv4_3 = self.conv_layer(self.conv4_2,name="conv4_3",load_weights = True)
		self.pool4 = self.max_pool(self.conv4_3,name = "pool4")

		self.conv5_1 = self.conv_layer(self.pool4,name="conv5_1",load_weights = True)
		self.conv5_2 = self.conv_layer(self.conv5_1,name="conv5_2",load_weights = True)
		self.conv5_3 = self.conv_layer(self.conv5_2,name="conv5_3",load_weights = True)
		self.pool5 = self.max_pool(self.conv5_3,name = "pool5")

		self.flatten_layer_branch=tf.layers.Flatten()(self.pool5)
		self.fc6 = self.fc_layer(self.pool5,name = "fc6")

		if train:
			self.fc6=tf.nn.dropout(self.fc6,0.5)

		self.fc7 = self.fc_layer(self.fc6,"fc7")
		if train:
			self.fc7= tf.nn.dropout(self.fc7,0.5)
		
		if random_init_fc8:
			self.score_fr = self.score_layer(self.fc7,"score_fr",num_classes)
		else:
			self.score_fr  = self.fc_layer(self.fc7,"score_fr",num_classes=num_classes,relu=False)

		self.pred = tf.argmax(self.score_fr,dimension=3)

		self.upscore2 = self.upscore_layer(	self.score_fr,
											shape=tf.shape(self.pool4),
											num_classes = num_classes,
											debug=debug,
											name='upscore2',
											ksize=4,
											stride=2)

		self.score_pool4 = self.score_layer(self.pool4,name="score_pool4",num_classes=num_classes)
		self.fuse_pool4 = tf.add(self.upscore2,self.score_pool4)

		self.upscore4 = self.upscore_layer(	self.fuse_pool4,
											shape=tf.shape(self.pool3),
											num_classes = num_classes,
											debug=debug,
											name='upscore4',
											ksize=4,
											stride=2)


		self.score_pool3 = self.score_layer(self.pool3,name="score_pool3",num_classes=num_classes)
		self.fuse_pool3= tf.add(self.upscore4,self.score_pool3)


		self.upscore32 = self.upscore_layer(	self.fuse_pool3,
											shape=tf.shape(self.bgr),
											num_classes = num_classes,
											debug=debug,
											name='upscore32',
											ksize=4,
											stride=2)

		self.pred_up = tf.argmax(self.upscore32,dimension=3)
		logits,decoded_logits = self.build_decoder(num_classes)

		return logits,decoded_logits





		# self.conv1_1 = self.conv_layer(bgr,filter_size=3,num_filters=64,stride=1,name='conv1_1')
		# conv1_2 = self.conv_layer(conv1_1,filter_size=3,num_filters=64,stride=1,name='conv1_2')
		# pool1 = self.max_pool(conv1_2,ksize = 2,stride=2,name = "pool1")
		
		# conv2_1 = self.conv_layer(pool1,filter_size=3,num_filters=128,stride=1,name='conv2_1')
		# conv2_2 = self.conv_layer(conv2_1,filter_size=3,num_filters=128,stride=1,name='conv2_2')
		# pool2 = self.max_pool(conv2_2,ksize = 2,stride=2,name = "pool2")
		
		# conv3_1 = self.conv_layer(pool2,filter_size=3,num_filters=256,stride=1,name='conv3_1')
		# conv3_2 = self.conv_layer(conv3_1,filter_size=3,num_filters=256,stride=1,name='conv3_2')
		# conv3_3 = self.conv_layer(conv3_2,filter_size=3,num_filters=256,stride=1,name='conv3_3')
		# pool3 = self.max_pool(conv3_3,ksize = 2,stride=2,name = "pool3")
		
		# conv4_1 = self.conv_layer(pool3,filter_size=3,num_filters=512,stride=1,name='conv4_1')
		# conv4_2 = self.conv_layer(conv4_1,filter_size=3,num_filters=512,stride=1,name='conv4_2')
		# conv4_3 = self.conv_layer(conv4_2,filter_size=3,num_filters=512,stride=1,name='conv4_3')
		# pool4 = self.max_pool(conv4_3,ksize = 2,stride=2,name = "pool4")
		
		# conv5_1 = self.conv_layer(pool4,filter_size=3,num_filters=512,stride=1,name='conv5_1')
		# conv5_2 = self.conv_layer(conv5_1,filter_size=3,num_filters=512,stride=1,name='conv5_2')
		# conv5_3 = self.conv_layer(conv5_2,filter_size=3,num_filters=512,stride=1,name='conv5_3')
		# pool5 = self.max_pool(conv5_3,ksize = 2,stride=2,name = "pool5")
		
		# layer_flat = self.flatten_layer(pool5)
		
		# fc1 = self.fc_layer(layer_flat,4096,"FC1",use_relu=True)
		
		# fc2 = self.fc_layer(fc1,1024,"FC2",use_relu=True)
		
		# fc3 = self.fc_layer(fc2,128,"FC3",use_relu=True)
		
		# output_logits = self.fc_layer(fc3,1,"OUT",use_relu=False)
		
		#return output_logits
	


	def Compute_cross_entropy_mean(self,labels,softmax):

		cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), [3,3,3,3,1]),
								   reduction_indices=[1])

		cross_entropy_mean = tf.reduce_mean(cross_entropy,
											name='xentropy_mean')
		return cross_entropy_mean

	def loss(self,decoded_logits,labels,curve):

		logits_base = decoded_logits['logits']
		# fc_base = decoded_logits['fc_5_1']
		with tf.name_scope('loss'):
			n_c = logits_base.get_shape()[3]
			logits = tf.reshape(logits_base,(-1,n_c))
			shape = [logits.get_shape()[0],n_c]
			epsilon = tf.constant(value=0.000000001)

			labels = tf.to_float(tf.reshape(labels,(-1,n_c)))

			softmax = tf.nn.softmax(logits) + epsilon

			print ("_____LOSSSS______")
			print (shape)
			print (logits)
			print (labels)

			print ("____END LOSS______")
			cross_entropy_mean = self.Compute_cross_entropy_mean(labels,softmax)

			reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES

			weight_loss = tf.add_n(tf.get_collection(reg_loss_col),name = 'reg_loss')
			print ("------------------------------")
			print (decoded_logits['fc_5_1'])
			print (curve[:,0,:])
			print ("------------------------------")

			error_multi_factor=100

			loss_flatten_1=tf.reduce_mean(tf.log(tf.losses.mean_squared_error(labels=curve[:,0,:],predictions=decoded_logits['fc_5_1'])*error_multi_factor,name='FC_loss_1'))
			loss_flatten_2=tf.reduce_mean(tf.log(tf.losses.mean_squared_error(labels=curve[:,1,:],predictions=decoded_logits['fc_5_2'])*error_multi_factor,name='FC_loss_2'))
			loss_flatten_3=tf.reduce_mean(tf.log(tf.losses.mean_squared_error(labels=curve[:,2,:],predictions=decoded_logits['fc_5_3'])*error_multi_factor,name='FC_loss_3'))
			loss_flatten_4=tf.reduce_mean(tf.log(tf.losses.mean_squared_error(labels=curve[:,3,:],predictions=decoded_logits['fc_5_4'])*error_multi_factor,name='FC_loss_4'))


			# loss_flatten_1=tf.reduce_mean(tf.losses.mean_squared_error(labels=curve[:,0,:],predictions=decoded_logits['fc_5_1']),name='FC_loss_1')
			# loss_flatten_2=tf.reduce_mean(tf.losses.mean_squared_error(labels=curve[:,1,:],predictions=decoded_logits['fc_5_2']),name='FC_loss_2')
			# loss_flatten_3=tf.reduce_mean(tf.losses.mean_squared_error(labels=curve[:,2,:],predictions=decoded_logits['fc_5_3']),name='FC_loss_3')
			# loss_flatten_4=tf.reduce_mean(tf.losses.mean_squared_error(labels=curve[:,3,:],predictions=decoded_logits['fc_5_4']),name='FC_loss_4')
			total_flat_loss=loss_flatten_1+loss_flatten_2+loss_flatten_3+loss_flatten_4
			
			total_loss= cross_entropy_mean + weight_loss +total_flat_loss

			losses = {}

			losses['total_loss'] = total_loss
			losses['xentropy'] = cross_entropy_mean
			losses['weight_loss'] = weight_loss


		return losses

	def optimizer(self,losses,global_step,learning_rate):
		hypes={}
		hypes['tensor'] = {}
		hypes['tensor']['global_step']=global_step
		total_loss=losses['total_loss']

		with tf.name_scope('training'):

			opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
										epsilon=0.00001)

			hypes['opt'] = opt

			grads_and_vars=opt.compute_gradients(total_loss)

			if True:
				grads,tvars = zip(*grads_and_vars)
				clipped_grads, norm = tf.clip_by_global_norm(grads,1.0)
				grads_and_vars = zip(clipped_grads, tvars)

			train_op = opt.apply_gradients(grads_and_vars,global_step=global_step)

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

			with tf.control_dependencies(update_ops):
				train_op = opt.apply_gradients(grads_and_vars,global_step=global_step)

			return train_op

	def evaluation(self,labels, decoded_logits, losses):
		eval_list = []
		logits = tf.reshape(decoded_logits['logits'], (-1, 2))
		labels = tf.to_int32(tf.reshape(labels, (-1, 2)))

		pred = tf.argmax(logits, dimension=1)

		negativ = tf.to_int32(tf.equal(pred, 0))
		tn = tf.reduce_sum(negativ*labels[:, 0])
		fn = tf.reduce_sum(negativ*labels[:, 1])

		positive = tf.to_int32(tf.equal(pred, 1))
		tp = tf.reduce_sum(positive*labels[:, 1])
		fp = tf.reduce_sum(positive*labels[:, 0])

		eval_list.append(('Acc. ', (tn+tp)/(tn + fn + tp + fp)))
		eval_list.append(('xentropy', losses['xentropy']))
		eval_list.append(('weight_loss', losses['weight_loss']))

		return eval_list





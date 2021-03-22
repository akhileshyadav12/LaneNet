from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import logging
from math import ceil
import sys
import lanenet_discriminative_loss

import numpy as np
class ENet(object):
	"""docstring for ClassName"""
	def __init__(self):
		# super(ClassName, self).__init__()
		# self.arg = arg

		self.wd=5e-4
		pass

	def weight_variable(self,shape):
		print ("Weight shape : ",shape)
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
		initial = tf.constant(value=0.0,shape=shape,dtype=tf.float32)
		return tf.get_variable('b',
							   dtype=tf.float32,
							   initializer=initial)


	def get_conv_filter(self,name,shape):
		init =tf.contrib.layers.xavier_initializer(
					uniform=True,
					seed=25,
					# shape=shape,
					dtype=tf.float32
				)
		# shape = self.data_dict[name][0].shape
		# print('Layer name:%s'%name)
		print('Weights/Filter shape: ',shape)
		var = tf.get_variable(name='filter_W',initializer=init,shape=shape)
		if not tf.get_variable_scope().reuse:
			weight_Decay = tf.multiply(tf.nn.l2_loss(var),self.wd,
				name='weight_loss')
			# print ("weight_decay: L2 Regularization Applied")
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weight_Decay)
		# __variable_summary(var)
		return var


	def batch_normalization(self,x,phase_train=False):
		with tf.variable_scope('bn'):
			n_out = x.get_shape()[3].value
			beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
										 name='beta', trainable=True)
			gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
										  name='gamma', trainable=True)
			batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
			ema = tf.train.ExponentialMovingAverage(decay=0.2)

			def mean_var_with_update():
				ema_apply_op = ema.apply([batch_mean, batch_var])
				with tf.control_dependencies([ema_apply_op]):
					return tf.identity(batch_mean), tf.identity(batch_var)

			mean, var = tf.cond(tf.constant(phase_train,dtype=tf.bool),
								mean_var_with_update,
								lambda: (ema.average(batch_mean), ema.average(batch_var)))
			normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
		return normed

	def max_pool(self,x,name,ksize_,stride,):
		# ksize_=2
		# stride=2
		pool=tf.nn.max_pool(x,
							  ksize=[1,ksize_,ksize_,1],
							  strides=[1,stride,stride,1],
							  padding="SAME",
							  name=name)
		print (pool)
		return pool

	def conv_layer(self,x,name,filt,out_depth,batch_norm,stride,PRelu,phase_train,rate=1,use_bias=True):

		with tf.variable_scope(name):
			with tf.variable_scope(name) as scope:

				prev_depth=x.get_shape().as_list()[3]
				if "int" in str(type(filt)) :
					shape=[filt,filt,prev_depth,out_depth]
				elif "list" in str(type(filt)):
					shape=[filt[0],filt[1],prev_depth,out_depth]
				filt = self.get_conv_filter(name,shape)
				print (x)
				# print ("debug:---------",filt,filt[1:])
				conv = tf.nn.conv2d(x,filt,strides=[1,stride,stride,1],dilations=[1,rate,rate,1],padding='SAME')
				if use_bias:
					conv_biases = self.bias_variable([shape[-1]])
					bias = tf.nn.bias_add(conv,conv_biases)
				else:
					bias=conv
				if batch_norm:
					bias=self.batch_normalization(bias,phase_train)
				if PRelu:
					bias = tf.nn.leaky_relu(bias,alpha=0.5)
				return bias




	def spatial_dropout(self,x,keep_prob,batch_size,phase_train=False):
		if phase_train:
			rate= 1- keep_prob
			input_shape=x.get_shape().as_list()
			noise_shape=tf.constant(value=[batch_size,1,1,input_shape[3]])
			output=tf.nn.dropout(x,noise_shape=noise_shape,keep_prob=rate,seed=0,name="dropout")
			return output
		return x

	def initial_block(self,input_):
		'''
		ENet Initial Block
		13 filters and MAx pool
		'''
		with tf.variable_scope('initial_block_1'):
			init_1 = self.conv_layer(input_,
									name="init_conv",
									filt=3,
									out_depth=13,
									batch_norm=True,
									stride=2,
									PRelu=True,
									phase_train=True)
			init_pool = self.max_pool(input_,name='init_pool',ksize_=2,stride=2)

			init_out = tf.concat([init_1,init_pool],axis=3,name="init_concat")

		return init_out


	def unpool(self,updates,mask,batch_size,k_size=[1,2,2,1],out_shape=None,scope='Unpool'):
		with tf.variable_scope(scope):
			mask=tf.cast(mask,tf.int32)
			input_shape=tf.shape(updates,out_type=tf.int32)

			if out_shape==None:
				out_shape=(input_shape[0],input_shape[1]*k_size[1],input_shape[2]*k_size[2],input_shape[3])
			if out_shape[0]==None:
				out_shape[0]=batch_size	
			one_like_mask=tf.ones_like(mask,dtype=tf.int32)
			batch_shape=tf.concat([[input_shape[0]],[1],[1],[1]],0)
			batch_range=tf.reshape(tf.range(out_shape[0],dtype=tf.int32),shape=batch_shape)
			b=one_like_mask*batch_range
			y=mask//(out_shape[2]*out_shape[3])
			x=(mask//out_shape[3])%out_shape[2]
			feature_range=tf.range(out_shape[3],dtype=tf.int32)
			f=one_like_mask*feature_range

			updates_size=tf.size(updates)
			indices=tf.transpose(tf.reshape(tf.stack([b,y,x,f]),[4,updates_size]))
			values=tf.reshape(updates,[updates_size])
			ret=tf.scatter_nd(indices,values,out_shape)
			return ret

	def bottleneck(self,inputs,
					output_depth,
					filter_size,
					batch_size,
					regularize_prob,
					projection_ratio=4,
					seed=0,
					is_training=True,
					downsampling=False,
					upsampling=False,
					pooling_indices=None,
					output_shape=None,
					dilated=False,
					dilation_rate=None,
					asymmetric=False,
					scope='bottleneck'
					):
		reduced_depth=int(inputs.get_shape().as_list()[3]/projection_ratio)
		with tf.variable_scope(scope):
			if downsampling:
				net_main,pooling_indices=tf.nn.max_pool_with_argmax(inputs,
																	ksize=[1,2,2,1],
																	strides=[1,2,2,1],
																	padding='SAME',
																	name="main_max_pool")

				
				inputs_shape=inputs.get_shape().as_list()
				depth_to_pad=abs(inputs_shape[3]-output_depth)
				paddings=tf.convert_to_tensor([[0,0],[0,0],[0,0],[0,depth_to_pad]])
				net_main=tf.pad(net_main,paddings=paddings,name="main_padding")


				#sub Branch
				#first projection 2x2 kernel and stride 2

				net=self.conv_layer(inputs,filt=2,out_depth=reduced_depth,stride=2,name="Conv_1",
									phase_train=is_training,batch_norm=True,PRelu=True)
				#second Conv Block
				net=self.conv_layer(net,filt=filter_size,out_depth=reduced_depth,stride=1,name="Conv_2",
									phase_train=is_training,batch_norm=True,PRelu=True)
				#final Projection
				net=self.conv_layer(net,filt=1,out_depth=output_depth,stride=1,name="Conv_3",
									phase_train=is_training,batch_norm=True,PRelu=True)
				#Regularizer
				net=self.spatial_dropout(net,batch_size=batch_size,phase_train=is_training,keep_prob=regularize_prob)

				#combine 2 branches
				net=tf.add(net,net_main,name="_add")
				net=tf.nn.leaky_relu(net,alpha=0.5)

				return net, pooling_indices,inputs_shape

			elif dilated:
				if not dilation_rate:
					raise ValueError('Dilation Rate is not given')
				net_main=inputs

				#sub Branch
				#first projection 2x2 kernel and stride 2

				net=self.conv_layer(inputs,filt=1,out_depth=reduced_depth,stride=1,name="Conv_1",
									phase_train=is_training,batch_norm=True,PRelu=True)
				#second Conv Block
				net=self.conv_layer(net,filt=filter_size,out_depth=reduced_depth,stride=1,name="Dilated_Conv_2",rate=dilation_rate,
									phase_train=is_training,batch_norm=True,PRelu=True)
				#final Projection
				net=self.conv_layer(net,filt=1,out_depth=output_depth,stride=1,name="Conv_3",
									phase_train=is_training,batch_norm=True,PRelu=True)
				#Regularizer
				net=self.spatial_dropout(net,batch_size=batch_size,phase_train=is_training,keep_prob=regularize_prob)
				net=tf.nn.leaky_relu(net,alpha=0.5)
				#combine 2 branches
				net=tf.add(net,net_main,name="_add_dilated")
				net=tf.nn.leaky_relu(net,alpha=0.5)

				return net#, pooling_indices,inputs_shape
			elif asymmetric:
				net_main=inputs

				#sub Branch
				#first projection 2x2 kernel and stride 2

				net=self.conv_layer(inputs,filt=1,out_depth=reduced_depth,stride=1,name="Conv_1",
									phase_train=is_training,batch_norm=True,PRelu=True)
				#second Conv Block
				net=self.conv_layer(net,filt=[filter_size,1],out_depth=reduced_depth,stride=1,name="Asymmetric_Conv_2a",
									phase_train=is_training,batch_norm=False,PRelu=False)
				net=self.conv_layer(net,filt=[1,filter_size],out_depth=reduced_depth,stride=1,name="Asymmetric_Conv_2b",
									phase_train=is_training,batch_norm=True,PRelu=True)
				#final Projection
				net=self.conv_layer(net,filt=1,out_depth=output_depth,stride=1,name="Conv_3",
									phase_train=is_training,batch_norm=True,PRelu=True)
				#Regularizer
				net=self.spatial_dropout(net,batch_size=batch_size,phase_train=is_training,keep_prob=regularize_prob)
				net=tf.nn.leaky_relu(net,alpha=0.5)
				#combine 2 branches
				net=tf.add(net,net_main,name="_add_dilated")
				net=tf.nn.leaky_relu(net,alpha=0.5)

				return net#, pooling_indices,inputs_shape

			elif upsampling:
				if pooling_indices==None:
					raise ValueError('Pooling Indices are not given')
				if output_shape==None:
					raise ValueError('Output Shape is not given')

				# Main Branch
				net_unpool=self.conv_layer(inputs,filt=1,out_depth=output_depth,stride=1,name="Main_Conv_1",
									phase_train=is_training,batch_norm=True,PRelu=False)
				net_unpool= self.unpool(net_unpool,pooling_indices,batch_size=batch_size,out_shape=output_shape,scope="unpool")
				 # Sub Branch

				net=self.conv_layer(inputs,filt=1,out_depth=reduced_depth,stride=1,name="Conv_1",
									phase_train=is_training,batch_norm=True,PRelu=True)
				#second Conv Block
				net_unpool_shape=net_unpool.get_shape().as_list()
				output_shape=[net_unpool_shape[0],net_unpool_shape[1],
								net_unpool_shape[2],reduced_depth]
				output_shape=tf.convert_to_tensor(output_shape)

				filter_size=[filter_size,filter_size,reduced_depth,reduced_depth]
				initer_=tf.contrib.layers.xavier_initializer(
					uniform=True,
					seed=None,
					# shape=shape,
					dtype=tf.float32
				)
				filters=tf.get_variable(shape=filter_size,initializer=initer_,dtype=tf.float32,
										name="Conv2D_Transpose_filter")

				net = tf.nn.conv2d_transpose(net,filter=filters,strides=[1,2,2,1],
											output_shape=output_shape,name="Conv2D_Transpose")

				net=self.batch_normalization(net,phase_train=is_training)
				relu = tf.nn.leaky_relu(net,alpha=0.5)
				#final Projection


				net=self.conv_layer(net,filt=1,out_depth=output_depth,stride=1,name="Conv_3",
									phase_train=is_training,batch_norm=True,PRelu=True)
				#Regularizer
				net=self.spatial_dropout(net,batch_size=batch_size,phase_train=is_training,keep_prob=regularize_prob)
				net=tf.nn.leaky_relu(net,alpha=0.5)
				#combine 2 branches
				net=tf.add(net,net_unpool,name="_add_upsample")
				net=tf.nn.leaky_relu(net,alpha=0.5)

				return net


			#Other Wise


			net_main=inputs

			#sub Branch
			#first projection 2x2 kernel and stride 2

			net=self.conv_layer(inputs,filt=1,out_depth=reduced_depth,stride=1,name="Conv_1",
								phase_train=is_training,batch_norm=True,PRelu=True)
			#second Conv Block
			net=self.conv_layer(net,filt=filter_size,out_depth=reduced_depth,stride=1,name="Conv_2",
								phase_train=is_training,batch_norm=True,PRelu=True)
			#final Projection
			net=self.conv_layer(net,filt=1,out_depth=output_depth,stride=1,name="Conv_3",
								phase_train=is_training,batch_norm=True,PRelu=True)
			#Regularizer
			net=self.spatial_dropout(net,batch_size=batch_size,phase_train=is_training,keep_prob=regularize_prob)
			net=tf.nn.leaky_relu(net,alpha=0.5)
			#combine 2 branches
			net=tf.add(net,net_main,name="_add_regular")
			net=tf.nn.leaky_relu(net,alpha=0.5)

			return net#, pooling_indices,inputs_shape




	def stage_2_(self,net,scope_value,is_training,batch_size):

		net=self.bottleneck(net,output_depth=128,filter_size=3,is_training=is_training,batch_size=batch_size,
				regularize_prob=0.1,
				scope='bottleneck_'+str(scope_value)+'_1')

		net=self.bottleneck(net,output_depth=128,filter_size=3,is_training=is_training,batch_size=batch_size,
			regularize_prob=0.1,
			dilated=True,dilation_rate=2, 
			scope='bottleneck_'+str(scope_value)+'_2')

		net=self.bottleneck(net,output_depth=128,filter_size=5,is_training=is_training,batch_size=batch_size,
			regularize_prob=0.1,
			asymmetric=True,
			scope='bottleneck_'+str(scope_value)+'_3')

		net=self.bottleneck(net,output_depth=128,filter_size=3,is_training=is_training,batch_size=batch_size,
			regularize_prob=0.1,
			dilated=True,dilation_rate=4, 
			scope='bottleneck_'+str(scope_value)+'_4')

		net=self.bottleneck(net,output_depth=128,filter_size=3,is_training=is_training,batch_size=batch_size,
			regularize_prob=0.1,
			scope='bottleneck_'+str(scope_value)+'_5')

		net=self.bottleneck(net,output_depth=128,filter_size=3,is_training=is_training,batch_size=batch_size,
			regularize_prob=0.1,
			dilated=True,dilation_rate=8, scope='bottleneck_'+str(scope_value)+'_6')

		net=self.bottleneck(net,output_depth=128,filter_size=5,is_training=is_training,batch_size=batch_size,
			regularize_prob=0.1,
			asymmetric=True,
			scope='bottleneck_'+str(scope_value)+'_7')

		net=self.bottleneck(net,output_depth=128,filter_size=3,is_training=is_training,batch_size=batch_size,
			regularize_prob=0.1,
			dilated=True,dilation_rate=16, 
			scope='bottleneck_'+str(scope_value)+'_8')

		return net

	def model(self,x,batch_size,skip_connection=True,
				stage_two_repeat=2,is_training=True,num_classes=4,
				num_features_instance=16,scope="ENet"):
		init_block = self.initial_block(x)

		output_logits={}
		if skip_connection:
			net_one=init_block
			# output_depth,
			# 		filter_size,
			# 		regularize_prob,
			# 		projection_ratio=4,
			# 		seed=0,
			# 		is_training=True,
			# 		downsampling=False,
			# 		upsampling=False,
			# 		pooling_indices=None,
			# 		output_shape=None,
			# 		dilated=False,
			# 		dilation_rate=None,
			# 		asymmetric=False,
			# 		scope='bottleneck'

		######_________________STAGE ONE___________________
		net,pooling_indices_1,input_shape_1=self.bottleneck(init_block,output_depth=64,filter_size=3,is_training=is_training,batch_size=batch_size,
			regularize_prob=0.01,downsampling=True,scope='bottleneck_1_0')
		net = self.bottleneck(net,output_depth=64,filter_size=3,regularize_prob=0.01,is_training=is_training,batch_size=batch_size,
						scope='bottleneck_1_1')
		net = self.bottleneck(net,output_depth=64,filter_size=3,regularize_prob=0.01,is_training=is_training,batch_size=batch_size,
						scope='bottleneck_1_2')
		net =self.bottleneck(net,output_depth=64,filter_size=3,regularize_prob=0.01,is_training=is_training,batch_size=batch_size,
						scope='bottleneck_1_3')
		net = self.bottleneck(net,output_depth=64,filter_size=3,regularize_prob=0.01,is_training=is_training,batch_size=batch_size,
						scope='bottleneck_1_4')

		if skip_connection:
			net_two=net

		net,pooling_indices_2,input_shape_2=self.bottleneck(net,output_depth=128,filter_size=3,is_training=is_training,batch_size=batch_size,
			regularize_prob=0.1,downsampling=True,scope='bottleneck_2_0')


		#Stage 2 and 3
		i=2
		net_2=self.stage_2_(net,is_training=is_training,batch_size=batch_size,scope_value="2")



		#Stage 4 
		with tf.name_scope('Binary_Seg'):
			i=3
			net=self.stage_2_(net_2,is_training=is_training,batch_size=batch_size,scope_value="bin_seg_3")

			bottleneck_scope="bottleneck_bin_seg"+str(i+1)
			net = self.bottleneck(net, output_depth=64, filter_size=3, upsampling=True,batch_size=batch_size,
							is_training=is_training,
							regularize_prob=0.1,
							pooling_indices=pooling_indices_2, 
							output_shape=input_shape_2, 
							scope=bottleneck_scope+'_1')

			if skip_connection:
				net = tf.add(net,net_two,name=bottleneck_scope+"_skip_connect_decode_1")

			net = self.bottleneck(net,output_depth=64,filter_size=3,regularize_prob=0.1,is_training=is_training,batch_size=batch_size,scope=bottleneck_scope+"_2")
			net = self.bottleneck(net,output_depth=64,filter_size=3,regularize_prob=0.1,is_training=is_training,batch_size=batch_size,scope=bottleneck_scope+"_3")

			#stage 5
			bottleneck_scope="bottleneck_bin_seg"+str(i+2)
			net = self.bottleneck(net, output_depth=16, filter_size=3, upsampling=True,batch_size=batch_size,
							is_training=is_training,
							regularize_prob=0.1,
							pooling_indices=pooling_indices_1, 
							output_shape=input_shape_1, 
							scope=bottleneck_scope+'_1')

			if skip_connection:
				net = tf.add(net,net_one,name=bottleneck_scope+"_skip_connect_decode_2")

			net =self. bottleneck(net,output_depth=16,filter_size=3,regularize_prob=0.1,is_training=is_training,batch_size=batch_size,scope=bottleneck_scope+"_2")
			# net = bottleneck(net,output_depth=64,filter_size=3,scope=bottleneck_scope+"_2")

			#final_ conv
			prev_inp_shape=net.get_shape()[3]
			filter_size=2
			filter_size=[filter_size,filter_size,num_classes,prev_inp_shape]

			initer_1=tf.contrib.layers.xavier_initializer(
						uniform=True,
						seed=None,
						# shape=shape,
						dtype=tf.float32
					)
			filters=tf.get_variable(shape=filter_size,initializer=initer_1,dtype=tf.float32,
									name="FULL_CONV_FILTER_bin_seg")

			output_logits['binary_seg_logits'] = tf.nn.conv2d_transpose(net,filter=filters,strides=[1,2,2,1],
										output_shape=[batch_size,480,640,num_classes],name="FULL_CONV_binary")
			# print (logits)
			output_logits['binary_seg_prob'] = tf.nn.softmax(output_logits['binary_seg_logits'], name='logits_to_softmax')





		with tf.name_scope('Instance_seg'):
			net=self.stage_2_(net_2,is_training=is_training,batch_size=batch_size,scope_value="instance_seg_3")
			bottleneck_scope="bottleneck_instance_seg"+str(i+1)
			net = self.bottleneck(net, output_depth=64, filter_size=3, upsampling=True,batch_size=batch_size,
							is_training=is_training,
							regularize_prob=0.1,
							pooling_indices=pooling_indices_2, 
							output_shape=input_shape_2, 
							scope=bottleneck_scope+'_1')

			if skip_connection:
				net = tf.add(net,net_two,name=bottleneck_scope+"_skip_connect_decode_1")

			net = self.bottleneck(net,output_depth=64,filter_size=3,regularize_prob=0.1,is_training=is_training,batch_size=batch_size,scope=bottleneck_scope+"_2")
			net = self.bottleneck(net,output_depth=64,filter_size=3,regularize_prob=0.1,is_training=is_training,batch_size=batch_size,scope=bottleneck_scope+"_3")

			#stage 5
			bottleneck_scope="bottleneck_instance_seg"+str(i+2)
			net = self.bottleneck(net, output_depth=16, filter_size=3, upsampling=True,batch_size=batch_size,
							is_training=is_training,
							regularize_prob=0.1,
							pooling_indices=pooling_indices_1, 
							output_shape=input_shape_1, 
							scope=bottleneck_scope+'_1')

			if skip_connection:
				net = tf.add(net,net_one,name=bottleneck_scope+"_skip_connect_decode_2")

			net =self. bottleneck(net,output_depth=16,filter_size=3,regularize_prob=0.1,is_training=is_training,batch_size=batch_size,scope=bottleneck_scope+"_2")
			# net = bottleneck(net,output_depth=64,filter_size=3,scope=bottleneck_scope+"_2")

			#final_ conv
			prev_inp_shape=net.get_shape()[3]
			filter_size=2
			filter_size=[filter_size,filter_size,num_features_instance,prev_inp_shape]

			initer_1=tf.contrib.layers.xavier_initializer(
						uniform=True,
						seed=None,
						# shape=shape,
						dtype=tf.float32
					)
			filters=tf.get_variable(shape=filter_size,initializer=initer_1,dtype=tf.float32,
									name="FULL_CONV_FILTER_ins_seg")

			output_logits['instance_seg_logits'] = tf.nn.conv2d_transpose(net,filter=filters,strides=[1,2,2,1],
										output_shape=[batch_size,480,640,num_features_instance],name="FULL_CONV_instance")
			# print (logits)
			# output_logits['binary_seg_prob'] = tf.nn.softmax(logits, name='logits_to_softmax')

		return output_logits



	def Compute_cross_entropy_mean(self,labels,softmax,loss_weights):
		# loss_weights = tf.reduce_sum(tf.multiply(onehot_labels, classes_weights), axis=3)
		cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), loss_weights),
								   reduction_indices=[1])

		cross_entropy_mean = tf.reduce_mean(cross_entropy,
											name='xentropy_mean')

		# loss = tf.losses.softmax_cross_entropy(
		# 	onehot_labels=onehot_labels,
		# 	logits=logits,
		# 	weights=loss_weights
		# )
		return cross_entropy_mean

	def instance_loss_(self,logits,y):
		train_features_dim=8

		pix_embedding=logits
		# normed = self.batch_normalization(logits,phase_train=True)
		# net=tf.nn.relu(features=normed, name="pix_embed_relu")
		# pix_embedding = self.conv_layer(net,
		# 							name="pix_embedding_conv",
		# 							filt=1,
		# 							out_depth=train_features_dim,
		# 							batch_norm=False,
		# 							stride=1,
		# 							PRelu=False,
		# 							phase_train=True,
		# 							use_bias=False)
		print(pix_embedding)
		pix_image_shape = (pix_embedding.get_shape().as_list()[1], 
						pix_embedding.get_shape().as_list()[2])

		instance_segmentation_loss, l_var, l_dist, l_reg = \
					lanenet_discriminative_loss.discriminative_loss(
						pix_embedding, y, train_features_dim,
						pix_image_shape, 0.5, 3.0, 1.0, 1.0, 0.001
					)

		return instance_segmentation_loss


	def loss(self,output_logits,y_bin,y_ins):
		logits_temp_bin = output_logits['binary_seg_logits']

		logits_temp_seg = output_logits['instance_seg_logits']


		# num_classes=4
		with tf.name_scope('loss'):


			# predictions = tf.argmax(probabilities,-1)

			# annotations= tf.argmax(y,-1)
			# print (predictions,y)
			# accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, annotations)
			# mean_IOU, mean_IOU_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions, labels=annotations, num_classes=num_classes)
			# per_class_accuracy, per_class_accuracy_update = tf.metrics.mean_per_class_accuracy(labels=annotations, predictions=predictions, num_classes=num_classes)
			# metrics_op = tf.group(accuracy_update, mean_IOU_update, per_class_accuracy_update)
			n_c=logits_temp_bin.get_shape()[3]
			index_array = tf.range(start=0,limit=n_c,delta=1)
			# mul_mat = tf.cast(mul_mat,dtype=tf.float32)

			y_bin_temp=tf.reshape(y_bin,shape=[-1,n_c])
			# y_bin_temp=tf.multiply(y_bin_temp,mul_mat)

			# print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$TEMP:",y_bin_temp)
			lables_plain = tf.reshape(tf.argmax(y_bin_temp,axis=1),shape=[-1])

			unique_labels, unique_id, counts = tf.unique_with_counts(lables_plain)
			weights_combine=[unique_labels, unique_id, counts]

			counts = tf.cast(counts, tf.float32)
			inverse_weights = tf.divide(
					1.0,
					tf.log(tf.add(tf.divide(counts, tf.reduce_sum(counts)), tf.constant(1.02)))
				)
			inverse_weights=tf.gather(inverse_weights,unique_labels)
			logits=tf.reshape(logits_temp_bin,(-1,n_c))
			shape=[logits.get_shape()[0],n_c]
			epsilon=tf.constant(value=0.0000001)

			labels=tf.to_float(tf.reshape(y_bin,(-1,n_c)))

			softmax = tf.nn.softmax(logits)+epsilon
			print ("_____LOSSSS______")
			print (shape)
			print (logits)
			print (labels)

			print ("____END LOSS______")

			cross_entropy_mean=self.Compute_cross_entropy_mean(labels,softmax,inverse_weights)
			reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES
			weight_loss = tf.add_n(tf.get_collection(reg_loss_col),name = 'reg_loss')


			instance_loss=self.instance_loss_(logits_temp_seg,y_ins)

			total_loss=cross_entropy_mean+weight_loss+instance_loss#+(1-accuracy)+mean_IOU

			losses={}
			losses['total_loss']=total_loss
			losses['xentropy']=cross_entropy_mean
			losses['weight_loss']=weight_loss
			losses['instance_loss']=instance_loss
			losses['weights_of_loss']=inverse_weights
			# losses['weights_combined']=weights_combine
			# losses['Others_1']=mul_mat
			# losses['Others_2']=y_bin_temp
			# losses['Others_3']=lables_plain
			# losses['acc']=accuracy
			# losses['mean_iou']=mean_IOU

			return losses#,accuracy,mean_IOU


	def optimizer(self,losses,global_step,learning_rate = 1e-5):
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
				# grads_and_vars = zip(clipped_grads, tvars)
				grads_and_vars = list(zip(clipped_grads, tvars))

			train_op = opt.apply_gradients(grads_and_vars,global_step=global_step)

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

			with tf.control_dependencies(update_ops):
				train_op = opt.apply_gradients(grads_and_vars,global_step=global_step)

			return train_op


	def evaluation(self,labels, logits, losses,num_classes):
		eval_list = []
		logits = tf.reshape(logits, (-1, num_classes))
		labels = tf.to_int32(tf.reshape(labels, (-1, num_classes)))

		pred = tf.argmax(logits, dimension=1)

		negativ = tf.to_int32(tf.equal(pred, 0))
		tn = tf.reduce_sum(negativ*labels[:, 0])
		fn = tf.reduce_sum(negativ*labels[:, 1])

		positive = tf.to_int32(tf.equal(pred, 1))
		tp = tf.reduce_sum(positive*labels[:, 1])
		fp = tf.reduce_sum(positive*labels[:, 0])

		eval_list.append(('Accuracy_. ', (tn+tp)/(tn + fn + tp + fp)))
		eval_list.append(('xentropy_', losses['xentropy']))
		eval_list.append(('weight_loss_', losses['weight_loss']))

		return eval_list

		
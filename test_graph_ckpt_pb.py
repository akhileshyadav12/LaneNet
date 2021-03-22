import tensorflow as tf 
import numpy as np 
# import matplotlib.pyplot as plt 
# import data_process as dp
import argparse
# Importing Dataset
from ENet.enet import ENet 
# import utils as utils
# import ENet as network
import logging
import cv2
import time
import json
from datetime import datetime

def main():
	print (":::::::::::::::::::::Training Module:::::::::::::::::::::::::::")
	parser = argparse.ArgumentParser(description='Lane Net')
	
	parser.add_argument('-d',help='data directory',			dest='data_dir',          
						type=str, 	default='../../../../dataset/carla/scenario/data_6.txt')

	parser.add_argument('-i',help='Image directory',		dest='img_dir',
						type=str,	default='/home/docker_share/dataset/carla/scenario/')  #dataset_store/lanes_5_color
	
	parser.add_argument('-t', help='test size fraction',	dest='test_size',         
						type=float, default=0.1)

	parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         
						type=float, default=0.5)

	parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          
						type=int,   default=10)

	parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', 
						type=int,   default=2000)

	parser.add_argument('-b', help='batch size',            dest='batch_size',        
						type=int,   default=8)

	parser.add_argument('-l', help='learning rate',         dest='learning_rate',     
						type=float, default=1.0e-4)
	parser.add_argument('-r', help='restore flag',         dest='restore',     
						type=bool, default=False)

	args = parser.parse_args()
	_ =vars(args)
	
	for i in _.keys():
		print (str(i)+" : "+str(_[i]))


	# before=0
	# after=80
	# X=X[before:after]
	# print X
	# Bin_Y=Bin_Y[before:after]
	# Ins_Y=Ins_Y[before:after]
	# print (X)

	# X=X[:40]
	# Y=Y[:40]
	# print (X)

	img_h=480
	img_w= 640 #28x28
	img_flat_size = img_h*(img_w)
	n_channels = 3
	n_classes = 2
	logs_path = "log/"
	lr=0.001
	epochs=300
	batch_size=1
	display_freq = 20
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		with tf.name_scope('Input'):
			x = tf.placeholder(tf.float32,shape=[None,img_h,img_w,n_channels],name='X')
			bin_y = tf.placeholder(tf.float32,shape=[None,img_h,img_w,n_classes],name='Bin_Y')
			ins_y = tf.placeholder(tf.float32,shape=[None,img_h,img_w,1],name='Ins_Y')

		network= ENet()
		output_logits = network.model(x,skip_connection=True,
						batch_size=batch_size,stage_two_repeat=2,
						is_training=False,num_features_instance=8,
						num_classes=n_classes,scope="ENet")

		init = tf.global_variables_initializer()
		init_l = tf.local_variables_initializer()
		merged = tf.summary.merge_all()
		

		# sess = tf.Session(config=config)
		sess.run(init)
		sess.run(init_l)
		global_step = 0
		summary_writer = tf.summary.FileWriter(logs_path,sess.graph)
		
		saver = tf.train.Saver()


		restore_epoch=495

		"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
		output_node_names=['Input/X','Binary_Seg/logits_to_softmax','Instance_seg/FULL_CONV_instance']
		temp_file_RUNS =  "RUNS/model_"+str(restore_epoch)+".ckpt"
		output_name = "RUNS/model_"+str(restore_epoch)+"_v1_1_new.pb"
		saver.restore(sess, temp_file_RUNS)
		print ("-----------------General Epoch Restored-----------------")
		"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
		graph_def=tf.get_default_graph().as_graph_def()
		node_list=[n.name for n in graph_def.node]
		for i in node_list:
    			if "logits_to_softmax" in i:
    					print (i)
		for node in graph_def.node:
			if node.op == 'RefSwitch':
				node.op = 'Switch'
				# print ("True")
				# print (node.name, node.op)
				for index in range(len(node.input)):
					if 'moving_' in node.input[index]:
						node.input[index] = node.input[index] + '/read'
			elif node.op == 'AssignSub':
				node.op = 'Sub'
				if 'use_locking' in node.attr: del node.attr['use_locking']

			elif node.op == 'AssignAdd':
				node.op = 'Add'
				if 'use_locking' in node.attr: del node.attr['use_locking']
		# print (node_list[:10])
		frozen_graph_def=tf.graph_util.convert_variables_to_constants(sess,graph_def,output_node_names)
		print (output_name)
		with open(output_name,'wb')as f:
			f.write(frozen_graph_def.SerializeToString())
		sess.close()
	
if __name__ == '__main__':
	main()
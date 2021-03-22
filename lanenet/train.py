# import tensorflow as tf 
import tensorflow as tf
# tf.disable_v2_behavior()

import numpy as np 
import matplotlib.pyplot as plt 
import data_process as dp
import argparse
# Importing Dataset
from ENet.enet import ENet 
import utils as utils
# import ENet as network
import logging
import cv2
import time
import json
from datetime import datetime
from config import global_config

def main():
	print (":::::::::::::::::::::Training Module:::::::::::::::::::::::::::")
	# parser = argparse.ArgumentParser(description='Lane Net')
	
	# parser.add_argument('-d',help='data directory',			dest='data_dir',          
	# 					type=str, 	default='../../../../dataset/carla/Town04_multilane/data.txt')

	# parser.add_argument('-i',help='Image directory',		dest='img_dir',
	# 					type=str,	default='/home/docker_share/dataset/carla/')  #dataset_store/lanes_5_color
	
	# parser.add_argument('-t', help='test size fraction',	dest='test_size',         
	# 					type=float, default=0.1)

	# parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         
	# 					type=float, default=0.5)

	# parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          
	# 					type=int,   default=10)

	# parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', 
	# 					type=int,   default=2000)

	# parser.add_argument('-b', help='batch size',            dest='batch_size',        
	# 					type=int,   default=8)

	# parser.add_argument('-l', help='learning rate',         dest='learning_rate',     
	# 					type=float, default=1.0e-4)
	# parser.add_argument('-r', help='restore flag',         dest='restore',     
	# 					type=bool, default=False)

	# args = parser.parse_args()
	# _ =vars(args)
	
	# for i in _.keys():
	# 	print (str(i)+" : "+str(_[i]))

	CFG = global_config.cfg
	X,Bin_Y,Ins_Y=dp.read_pandas_txt(CFG.TRAIN.DATA_DIR)
	# print (X)
	X,Bin_Y,Ins_Y = dp.randomize(X,Bin_Y,Ins_Y,is_list=True)

	# X=X[:40]
	# Y=Y[:40]
	# print (X)

	img_h= CFG.TRAIN.IMG_HEIGHT
	img_w= CFG.TRAIN.IMG_WIDTH #28x28
	img_flat_size = img_h*(img_w)
	# n_channels = 3
	# n_classes = 2
	# logs_path = "log/"
	# lr=0.01
	# epochs=1402
	# batch_size=8
	# display_freq = 20
	dataset=utils.image_data_read(X,Bin_Y,Ins_Y,CFG.TRAIN.IMG_DIR,img_h,img_w)
	dataset = dataset.shuffle(buffer_size = 100)
	dataset = dataset.repeat(CFG.TRAIN.EPOCHS)
	dataset = dataset.batch(CFG.TRAIN.BATCH_SIZE)
	iterator = dataset.make_one_shot_iterator()
	features = iterator.get_next()
	# iterator=iter(dataset)
	# features=next(iterator)

	with tf.name_scope('Input'):
		x = tf.placeholder(tf.float32,shape=[None,img_h,img_w,CFG.TRAIN.N_CHANNELS],name='X')
		bin_y = tf.placeholder(tf.float32,shape=[None,img_h,img_w,CFG.TRAIN.N_CLASSES],name='Bin_Y')
		ins_y = tf.placeholder(tf.float32,shape=[None,img_h,img_w,1],name='Ins_Y')

	network= ENet()
	output_logits = network.model(x,skip_connection=True,
					batch_size=CFG.TRAIN.BATCH_SIZE,stage_two_repeat=2,
					is_training=True,num_features_instance=8,
					num_classes=CFG.TRAIN.N_CLASSES,scope="ENet")

	with tf.name_scope("Loss"):
		losses = network.loss(output_logits,bin_y,ins_y)

	with tf.name_scope("Optimizer"):
		global_step = tf.Variable(0,trainable=False)
		train_op = network.optimizer(losses,global_step,learning_rate = 1e-5)
	
	# with tf.name_scope("Evaluation"):
	# 	# Add the Op to compare the logits to the labels during evaluation.
	# 	eval_list = network.evaluation(y, output_logits, losses,n_classes)
	# logits, decoded_logits = model.build_model(x,train=True,num_classes=n_classes,random_init_fc8=True)

	init = tf.global_variables_initializer()
	init_l = tf.local_variables_initializer()
	merged = tf.summary.merge_all()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	sess = tf.Session(config=config)
	sess.run(init)
	sess.run(init_l)
	global_step = 0
	summary_writer = tf.summary.FileWriter(CFG.TRAIN.LOG_PATH,sess.graph)
	num_tr_iter = int((Bin_Y.shape[0])/CFG.TRAIN.BATCH_SIZE)
	print ("NUM Iter:",num_tr_iter)
	saver = tf.train.Saver()

	epochs=CFG.TRAIN.EPOCHS
	restore=CFG.TRAIN.RESTORE_FLAG
	restore_epoch=CFG.TRAIN.RESTORE_EPOCH
	# restore_epoch=200

	all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
	print("Total Parameters:::::",sess.run(all_trainable_vars))
	if restore:

		temp_ =  "RUNS/model_"+str(restore_epoch)+".ckpt"
		saver.restore(sess, temp_)

	for epoch in range(epochs):
		if restore:
			if (epoch-5) < restore_epoch:
				continue
		# hours, rem = divmod(time.time(), 3600)
		# minutes, seconds = divmod(rem, 60)
		# print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours%60),int(minutes),seconds))
		print (datetime.now())
		print('Training Epoch: {}'.format(epoch+1))
		overall_loss=0.0
		for iteration in range(num_tr_iter):
			global_step+=1
			
			[x_batch_images,y_batch_labels,y_instance] = sess.run(features)
			feed_dict_batch = {x:x_batch_images,bin_y:y_batch_labels,ins_y:y_instance}
			_,loss_batch,out_,out_prob_= sess.run([train_op,losses,output_logits['binary_seg_prob'],
													output_logits['instance_seg_logits']],
													feed_dict=feed_dict_batch)
			overall_loss+=loss_batch['total_loss']
			if iteration % CFG.TRAIN.DISPLAY_STEP	 == 0:
				print("iter:",iteration," Loss= ",loss_batch,"\n")
			
		if epoch%1==0:

			# print('Training Epoch: {}'.format(epoch+1))
			# print "SHAPEPEE::",out_.shape
			for out_write_i in range(CFG.TRAIN.N_CLASSES):
				out11 = out_[:,:,:,out_write_i].reshape(CFG.TRAIN.BATCH_SIZE,img_h,img_w)
				
				out_pro = np.array((out11[0]>0.7)*255,dtype=np.uint8)
				write_name_img = "temp"+"_"+str(out_write_i)+".png"
				cv2.imwrite("RUNS/images/"+write_name_img,out_pro)
				# cv2.imshow("d",out_pro)
				# cv2.waitKey(10)
			
		#print ("iter {0:3d}:\t Loss={1:.3f}".format(iteration,loss_batch))
		overall_loss/=float(num_tr_iter)
		print ("AVERAGE LOSS:",overall_loss)
		if ((epoch%200==0 and epoch<50 )or(epoch%200==0)):	
			
			save_path = saver.save(sess, "RUNS/model_{0}.ckpt".format(epoch))
			print ("Model saved in path: %s" % save_path)

			for out_write_i in range(CFG.TRAIN.N_CLASSES):
				out11 = out_[:,:,:,out_write_i].reshape(CFG.TRAIN.BATCH_SIZE,img_h,img_w)
				
				out_pro = np.array((out11[0]>0.7)*255,dtype=np.uint8)
				write_name_img = "epoch_"+str(epoch)+"_"+str(out_write_i)+".png"
				cv2.imwrite("RUNS/images/"+write_name_img,out_pro)
				
	sess.close()
	
if __name__ == '__main__':
	main()
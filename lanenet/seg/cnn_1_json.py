import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import data_process_json as dp
import argparse
# Importing Dataset
from tensorflow.examples.tutorials.mnist import input_data
import utils_json as utils
import network_fcn_json as network
import logging
import cv2
import time
import json


if __name__ == '__main__':
	print ("ENTERED:::::::::::::::::::::::::::")
	parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
	parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='/home/docker_share/ros/car/')
	parser.add_argument('-i', help='Image directory',        dest='img_dir',          type=str,   default='/home/docker_share/ssd2/dataset/')  #dataset_store/lanes_5_color
	# parser.add_argument('-i', help='image directory',       dest='img_dir',           type=str,   default='/home/docker_share/ros/car/')
	parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
	parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
	parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
	parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=2000)
	parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=8)
	parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
	args = parser.parse_args()

	# x_train,y_train,x_valid,y_valid=dp.load_data(mode="train")
	# x_test,y_test = dp.load_data(mode="test")
	# print ("size of:")
	# print ("- Training -set :\t\tX:{},Y:{}".format((x_train.shape),len(y_train)))
	# print ("- Validation -set :\t\tX:{},Y:{}".format(len(x_valid),len(y_valid)))
	# print ("- Test -set :\t\t\tX:{},Y:{}".format(len(x_test),len(y_test)))
	
	# args.data_dir = "training_data_1.csv"
	# train_col_names = ['center']
	# test_col_names = ['steer']
	# data_extract_col_name = np.append(train_col_names,test_col_names)
	# # print data_extract_col_name
	# x,y = dp.read_pandas_csv(args.data_dir,data_extract_col_name,train_col_names,test_col_names)
	# x,y = dp.randomize(x,y,is_list=True)
	# x_train,y_train,x_valid,y_valid = dp.validation_split(ratio=0.70,x_t= x,y_t=y)


	args.data_dir = "../../ssd2/dataset/Town04_multilane/data.txt"
	args.json_dir = "../../ssd2/dataset/Town04_multilane/Town04_process.json"

	train_col_names = ['center']
	test_col_names = ['steer']
	data_extract_col_name = np.append(train_col_names,test_col_names)
	# print data_extract_col_name
	x,y,coords_vals = dp.read_pandas_txt(args.data_dir,args.json_dir,train=True)

	x,y,coords_vals = dp.randomize(x,y,coords_vals,is_list=True)

	coords_shape=coords_vals.shape
	# annot_json=dp.read_pandas_json(args.json_dir)
	# print x,y
	x_train,y_train,coords_train,x_valid,y_valid,coords_valid = dp.validation_split(ratio=1.0,x_t= x,y_t=y,coords_t=coords_vals)
	#print (x_train,y_train)
	# print (y_train.shape)
	
	
	
	img_h=270
	img_w= 480 #28x28
	img_flat_size = img_h*(img_w)
	n_channels = 3
	n_classes = 5
	logs_path = "log/"
	lr=0.001
	epochs=101
	batch_size=8
	display_freq = 20
	dataset=utils.image_data_read(x_train,y_train,args.img_dir,coords_train,img_h,img_w)
	dataset = dataset.shuffle(buffer_size = 100)
	dataset = dataset.repeat(epochs)
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_one_shot_iterator()
	features = iterator.get_next()
	# print features

	# while 1:
	# 	pass

	if 0:
		dataset_v=utils.image_data_read(x_valid,y_valid,coords_valid,args.img_dir,img_h,img_w)
		dataset_v = dataset_v.shuffle(buffer_size = 100)
		# dataset_v = dataset_v.repeat(epochs)
		dataset_v = dataset_v.batch(batch_size=y_valid.shape[0])
		iterator_v = dataset_v.make_one_shot_iterator()
		features_v = iterator_v.get_next()
	# print features
	# sess=tf.Session()
	# for _ in range(100):
	# 	# sess.run(iterator.initializer)
	# 	while True:
	# 		try:
	# 			[img_inp,label]=sess.run(features)
	# 			print img_inp.shape
	# 			print label.shape
	# 		except tf.errors.OutOfRangeError:
	# 			break

	# print x_train,y_train,x_valid,y_valid
	
	


	with tf.name_scope('Input'):
		x = tf.placeholder(tf.float32,shape=[None,img_h,img_w,n_channels],name='X')
		y = tf.placeholder(tf.float32,shape=[None,img_h,img_w,n_classes],name='Y')
		curve = tf.placeholder(tf.float32,shape=[None,coords_shape[1],coords_shape[2]],name='COORDS')
	print ("___________________________________________________________________________")
	model= network.FCN8VGG(vgg16_npy_path="vgg16.npy")
	logits, decoded_logits = model.build_model(x,train=True,num_classes=n_classes,random_init_fc8=True)

	with tf.name_scope("Loss"):
		losses = model.loss(decoded_logits,y,curve)

	with tf.name_scope("Optimizer"):
		global_step = tf.Variable(0,trainable=False)
		train_op = model.optimizer(losses,global_step,learning_rate = 1e-5)
	
	with tf.name_scope("Evaluation"):
		# Add the Op to compare the logits to the labels during evaluation.
		eval_list = model.evaluation(y, decoded_logits, losses)


	graph = {}
	graph['losses'] = losses
	graph['eval_list'] = eval_list
	graph['train_op'] = train_op
	graph['global_step'] = global_step
	graph['learning_rate'] = 1e-5
	graph['decoded_logits'] = 1e-5


	# print ("___________________________________________________________________________")
	# with tf.variable_scope('Train'):
	# 	with tf.variable_scope('Loss'):
	# 		loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y,predictions=output_logits),name='loss')
	# 	tf.summary.scalar('loss',loss)
	# 	with tf.variable_scope('Loss'):
	# 		optimizer = tf.train.AdamOptimizer(learning_rate=lr,name='Adam-op').minimize(loss)
		# with tf.variable_scope('Accuracy'):
		# 	correct_prediction = tf.equal(tf.argmax(output_logits,1),tf.argmax(y,1),name='correct_prediction')
		# 	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='accuracy')
		# tf.summary.scalar('accuracy',accuracy)
		# with tf.variable_scope("Prediction"):
		# 	cls_prediction=tf.argmax(output_logits,axis=1,name='prediction')

	init = tf.global_variables_initializer()
	merged = tf.summary.merge_all()
	sess = tf.Session()
	sess.run(init)
	global_step = 0
	summary_writer = tf.summary.FileWriter(logs_path,sess.graph)
	num_tr_iter = int((y_train.shape[0])/batch_size)
	print ("NUM Iter:",num_tr_iter)
	saver = tf.train.Saver()
	for epoch in range(epochs):

		hours, rem = divmod(time.time(), 3600)
		minutes, seconds = divmod(rem, 60)
		print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours%60),int(minutes),seconds))
		print('Training Epoch: {}'.format(epoch+1))
		overall_loss=0.0
		for iteration in range(num_tr_iter):
			global_step+=1
			
			[x_batch_images,y_batch_labels,coord_batch_labels] = sess.run(features)
			#y_batch_labels = y_batch_labels.reshape((y_batch_labels.shape[0],1))
			feed_dict_batch = {x:x_batch_images,y:y_batch_labels,curve:coord_batch_labels}
			# l1 = np.array(y_batch_labels[0,:,:,0].reshape(img_h,img_w)*255,dtype=np.uint8)
			# l2 = np.array(y_batch_labels[0,:,:,1].reshape(img_h,img_w)*255,dtype=np.uint8)
			# l3 = np.array(y_batch_labels[0,:,:,2].reshape(img_h,img_w)*255,dtype=np.uint8)
			# l4 = np.array(y_batch_labels[0,:,:,3].reshape(img_h,img_w)*255,dtype=np.uint8)
			# l5 = np.array(y_batch_labels[0,:,:,4].reshape(img_h,img_w)*255,dtype=np.uint8)

			'''------------------------Coord Check----------------------------
			inp_0=np.array(x_batch_images[0,:,:,:].reshape(img_h,img_w,3),dtype=np.uint8)
			inp_0=cv2.resize(inp_0,(1280,720))
			print np.array(coord_batch_labels[0])
			coords_temp=np.squeeze(np.array(coord_batch_labels[0]).reshape((4,16,2)))
			print coords_temp
			for temp_count in range(4):
				for temp_count_0 in range(15):
					cv2.line(inp_0,tuple(coords_temp[temp_count][temp_count_0]),tuple(coords_temp[temp_count][temp_count_0+1]),[255,0,255],thickness=3)
			# cv2.imshow("ld",l1)
			# cv2.imshow("ld1",l2)
			# cv2.imshow("ld2",l3)
			# cv2.imshow("ld3",l4)
			cv2.imshow("ld4",inp_0)
			cv2.waitKey(0)
			'''

			'''------------------------Coord Curve cooeff Check----------------------------'''
			inp_0=np.array(x_batch_images[0,:,:,:].reshape(img_h,img_w,3),dtype=np.uint8)
			inp_0=cv2.resize(inp_0,(1280,720))
			print np.array(coord_batch_labels[0])
			xp = np.linspace(0, 1280, 50)
			
			# coords_temp=np.squeeze(np.array(coord_batch_labels[0]).reshape((4,16,2)))
			# print curve_1
			curve_1=np.array(coord_batch_labels[0][0]).squeeze()
			curve_poly=np.poly1d(curve_1)
			yp=curve_poly(xp)
			curve_coords=np.squeeze(np.array([xp,yp],dtype=np.int32)).T
			for i in range(len(xp)-1):
				cv2.line(inp_0,tuple(curve_coords[i]),tuple(curve_coords[i+1]),[255,0,255],thickness=3)
			curve_1=np.array(coord_batch_labels[0][1]).squeeze()
			curve_poly=np.poly1d(curve_1)
			yp=curve_poly(xp)
			curve_coords=np.squeeze(np.array([xp,yp],dtype=np.int32)).T
			for i in range(len(xp)-1):
				cv2.line(inp_0,tuple(curve_coords[i]),tuple(curve_coords[i+1]),[255,0,255],thickness=3)
			curve_1=np.array(coord_batch_labels[0][2]).squeeze()
			curve_poly=np.poly1d(curve_1)
			yp=curve_poly(xp)
			curve_coords=np.squeeze(np.array([xp,yp],dtype=np.int32)).T
			for i in range(len(xp)-1):
				cv2.line(inp_0,tuple(curve_coords[i]),tuple(curve_coords[i+1]),[255,0,255],thickness=3)
			curve_1=np.array(coord_batch_labels[0][3]).squeeze()
			curve_poly=np.poly1d(curve_1)
			yp=curve_poly(xp)
			curve_coords=np.squeeze(np.array([xp,yp],dtype=np.int32)).T
			for i in range(len(xp)-1):
				cv2.line(inp_0,tuple(curve_coords[i]),tuple(curve_coords[i+1]),[255,0,255],thickness=3)
				
			# for temp_count in range(4):
			# 	for temp_count_0 in range(15):
			# 		cv2.line(inp_0,tuple(coords_temp[temp_count][temp_count_0]),tuple(coords_temp[temp_count][temp_count_0+1]),[255,0,255],thickness=3)
			# cv2.imshow("ld",l1)
			# cv2.imshow("ld1",l2)
			# cv2.imshow("ld2",l3)
			# cv2.imshow("ld3",l4)
			cv2.imshow("ld4",inp_0)
			cv2.waitKey(0)
			



			_,loss_batch,out_,out_curve = sess.run([train_op,losses,decoded_logits['softmax'],decoded_logits['fc_5_2']],feed_dict=feed_dict_batch)
			overall_loss+=loss_batch['total_loss']
			if iteration % display_freq == 0:
				
				# print np.array(out_).shape
				#summary_writer.add_summary(summary_tr,global_step)
				print("iter:",iteration," Loss= ",loss_batch,"\n")
				
				if epoch%20==0:
					# print('Training Epoch: {}'.format(epoch+1))
					
					out11 = out_[:,0].reshape(batch_size,img_h,img_w)
					out22 = out_[:,1].reshape(batch_size,img_h,img_w)
					out33 = out_[:,2].reshape(batch_size,img_h,img_w)
					out44 = out_[:,3].reshape(batch_size,img_h,img_w)
					out55 = out_[:,4].reshape(batch_size,img_h,img_w)
					
					out_pro = np.array((out11[0]>0.7)*255,dtype=np.uint8)
					out_pro1 = np.array((out22[0]>0.7)*255,dtype=np.uint8)
					out_pro2 = np.array((out33[0]>0.7)*255,dtype=np.uint8)
					out_pro3 = np.array((out44[0]>0.7)*255,dtype=np.uint8)
					out_pro4 = np.array((out55[0]>0.7)*255,dtype=np.uint8)
					
					# cv2.imshow("d",out_pro)
					# cv2.imshow("d1",out_pro1)
					# cv2.imshow("d2",out_pro2)
					# cv2.imshow("d3",out_pro3)
					# cv2.imshow("d4",out_pro4)
					# cv2.waitKey(10)
					
				#print ("iter {0:3d}:\t Loss={1:.3f}".format(iteration,loss_batch))

		overall_loss/=float(num_tr_iter)
		print ("AVERAGE LOSS:",overall_loss)
		if ((epoch%20==0 and epoch<50 )or(epoch%20==0)):		
			save_path = saver.save(sess, "RUNS/model_{0}.ckpt".format(epoch))
			print "out_curve:",out_curve
			print "out_curve_annot:",coord_batch_labels[:,1,:]
			print ("Model saved in path: %s" % save_path)
			write_name_img = "epoch_"+str(epoch)+"_0.png"
			cv2.imwrite("RUNS/images/"+write_name_img,out_pro)
			write_name_img = "epoch_"+str(epoch)+"_1.png"
			cv2.imwrite("RUNS/images/"+write_name_img,out_pro1)
			write_name_img = "epoch_"+str(epoch)+"_2.png"
			cv2.imwrite("RUNS/images/"+write_name_img,out_pro2)
			write_name_img = "epoch_"+str(epoch)+"_3.png"
			cv2.imwrite("RUNS/images/"+write_name_img,out_pro3)
			write_name_img = "epoch_"+str(epoch)+"_4.png"
			cv2.imwrite("RUNS/images/"+write_name_img,out_pro4)

		# [x_valid_images,y_valid_labels] = sess.run(features_v)
		# # print x_valid_images.shape,y_valid_labels
		# y_valid_labels = y_valid_labels.reshape((y_valid_labels.shape[0],1))
		# feed_dict_valid = {x:x_valid_images,y:y_valid_labels}
		# # loss_valid = sess.run(loss,feed_dict=feed_dict_valid)
		# print('-------------------------------------------------')
		# print ("Epoch:{0}, validation_loss:{1:.2f}".format(epoch+1,loss_valid))
		# print('-------------------------------------------------')

	
	


	# x_test,y_test = dp.load_data(mode="test")
	# feed_dict_test = {x:x_test,y:y_test}
	# loss_test = sess.run(loss,feed_dict=feed_dict_test)
	# print ('______________________________________________')
	# print ("Test loss: {0:.2f}".format(loss_test))
	# print ('______________________________________________')
	# cls_pred = sess.run(cls_prediction,feed_dict=feed_dict_test)
	# cls_true = np.argmax(y_test,axis=1)
	# dp.plot_images(x_test,cls_true,cls_pred,title='Correct Examples')
	# dp.plot_example_errors(x_test,cls_true,cls_pred,title='Misclassified Examples')
	# plt.show()
	sess.close()
	

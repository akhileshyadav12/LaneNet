import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
# import data_process as dp
import time
import argparse
from PIL import Image as Image_pil
# import network_fcn_json as network
from ENet.enet import ENet
# import data_process_json as dp
# from crf import DenseCRF
# import cv2
# Importing Dataset
# from tensorflow.examples.tutorials.mnist import input_data
import cv2
import lanenet_postprocess 
# import network
print (time.time())

class model_predict(object):
	def __init__(self):
		# x_test,y_test = dp.load_data(mode="test")
		# print ("- Test -set :\t\t\tX:{},Y:{}".format(len(x_test),len(y_test)))
		# model = network.Network()
		# data_dir = "../../dataset_store/lanes_5_color/"
		# data_dir = "dataset/"
		#video_path = "../../dataset_store/lanes_5_color/Upper Maurienne Valley - Indoor Cycling Training.mp4"
		# cam = cv2.VideoCapture(video_path)
		# x = dp.read_pandas_txt(data_dir+"train_1.txt",train=False)
		# tf.reset_default_graph()
		self.lane_color_1 =np.array([1,1,0],dtype=np.uint8)
		self.lane_color_2 =np.array([1,0,1],dtype=np.uint8)
		self.lane_color_3 =np.array([0,1,1],dtype=np.uint8)
		self.lane_color_4 =np.array([0,0,1],dtype=np.uint8)
		self.lane_color_5 =np.array([0,0,0],dtype=np.uint8)
		self.img_h=  270#270
		self.img_w = 480#480 #28x28
		self.n_channels = 3

		self.batch_size =1
	
		# self.postprocessor = self.setup_postprocessor()
		with tf.device('/device:GPU:0'):
			with tf.Graph().as_default():
				# Create placeholder for input

				# self.image_pl = tf.placeholder(tf.float32)
				# self.image = tf.expand_dims(self.image_pl, 0)
				img_h=270
				n_channels=3
				img_w= 480
				self.image_pl = tf.placeholder(tf.float32,shape=[None,img_h,img_w,n_channels])

				# build Tensorflow graph using the model from logdir
				network= ENet()
				# self.logits,self.probabilities = network.model(self.image_pl,skip_connection=True,
				# 	batch_size=self.batch_size,stage_two_repeat=2,is_training=False,num_classes=5,scope="ENet")
				self.output_logits = network.model(self.image_pl,skip_connection=True,
					batch_size=self.batch_size,stage_two_repeat=2,
					is_training=False,num_features_instance=64,
					num_classes=2,scope="LaneNet")
				# self.logits, self.decoded_logits = model.build_model(self.image_pl,train=False,num_classes=5,random_init_fc8=True)

				# logging.info("Graph build successfully.")

				# Create a session for running Ops on the Graph.
				#sess = tf.Session()
				# config = tf.ConfigProto()
				# config.gpu_options.allow_growth = True
				# config.gpu_options.per_process_gpu_memory_fraction = 0.4
				postprocessor = lanenet_postprocess.LaneNetPostProcessor()
				gpu_options = tf.GPUOptions(allow_growth=True)

				# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
				# init session
				self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
		
				self.saver = tf.train.Saver()
				self.saver.restore(self.sess, "/home/docker_share/ranjith_dev/lanenet/RUNS/model_1000.ckpt")
		#return sess
	def remove_sess(self):
		self.sess.close()
		cv2.destroyAllWindows()
		print ("Session Closexd::::::::::::::::::::::::::::::::::")

	# def setup_postprocessor(self):
	# 	# CRF post-processor
	# 	postprocessor = DenseCRF(
	# 		iter_max=5,
	# 		pos_xy_std=3,
	# 		pos_w=5,
	# 		bi_xy_std=67,
	# 		bi_rgb_std=3,
	# 		bi_w=4,
	# 	)
	# 	return postprocessor

	def predict(self,image):
		
		# _,frame = cam.read()
		# # for i in x:
		# for i in range(1000):
		# 	_,frame = cam.read()
		# while True:
		# if _:
		# _,frame = cam.read()
		# _,frame = cam.read()
		# _,frame = cam.read()
		# _,frame = cam.read()
		# _,frame = cam.read()
		# _,frame = cam.read()
		# _,frame = cam.read()
		# cv2.imshow("Frame",frame)
		# key = cv2.waitKey(1)
		# # print key
		# if key == ord("k"):
		# 	continue
		# 	# print i
		# # pth = data_dir+i
		# # pth = i
		# # img = Image_pil.open(pth)


		img_h=270#270
		n_channels=3
		img_w= 480#480
		image=np.array(image)
		img = np.array(image)
		# cv2.imshow("frea",img)
		img = cv2.resize(img,(self.img_w,self.img_h))
	
		#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		# cv2.waitKey(0)
		# img = img.resize((img_w,img_h))
		feed = {self.image_pl: [img]}
		# softmax = self.probabilities
		# curve_1 = self.decoded_logits['fc_5_1']
		# curve_2 = self.decoded_logits['fc_5_2']
		# curve_3 = self.decoded_logits['fc_5_3']
		# curve_4 = self.decoded_logits['fc_5_4']
		prev = time.time()

		out_values= self.sess.run([self.output_logits], feed_dict=feed)
		now_=time.time()

		print (now_-prev)
		postprocessor=postprocess_result = postprocessor.postprocess(
            binary_seg_result=out_values['binary_seg_logits'],
            instance_seg_result=out_values['instance_seg_logits'],
            source_image=img
        )
        mask_image = postprocess_result['mask_image']

		# Load weights from logdir
		#core.load_weights(logdir, sess, saver)
		# curve_1 = np.squeeze(np.array(coords_1,dtype=np.float32))
		# curve_2 = np.squeeze(np.array(coords_2,dtype=np.float32))
		# curve_3 = np.squeeze(np.array(coords_3,dtype=np.float32))
		# curve_4 = np.squeeze(np.array(coords_4,dtype=np.float32))
		# curve_1 = np.squeeze(np.array(coords_1))

		# print (np.squeeze(np.array(coords_1)).shape)
		# print (coords_1)
		
		# coord_1_1=np.reshape(curve_1,(16,2),dtype=np.float32)
		# print coord_1_1
		# print coord_1_1*np.array([1280.,720.])
		# print (np.reshape(curve_1,(16,2))*np.array([[1280.,720.]]))
		
		# if 1:
		# 	coords_1=np.array(np.reshape(curve_1,(16,2)),dtype=np.int32)#*np.array([1280.,720.])D
		# 	coords_1_test=np.array(np.reshape(curve_1,(16,2)),dtype=np.int32)
		# 	# print("Hiiiiiii")
		# 	# print(coords_1_test)
		# 	# print (coords_1)
		# 	coords_temp=coords_1
		# 	for i in range(15):
		# 		cv2.line(image,tuple(coords_temp[i]),tuple(coords_temp[i+1]),[255,0,255],thickness=3)
		# 	coords_1=np.array(np.reshape(curve_2,(16,2)),dtype=np.int32) #*np.array([1280.,720.]
		# 	coords_temp=coords_1
		# 	for i in range(15):
		# 		cv2.line(image,tuple(coords_temp[i]),tuple(coords_temp[i+1]),[0,0,255],thickness=3)
		# 	coords_1=np.array(np.reshape(curve_3,(16,2)),dtype=np.int32)
		# 	coords_temp=coords_1
		# 	for i in range(15):
		# 		cv2.line(image,tuple(coords_temp[i]),tuple(coords_temp[i+1]),[255,0,0],thickness=3)
		# 	coords_1=np.array(np.reshape(curve_4,(16,2)),dtype=np.int32)
		# 	coords_temp=coords_1
		# 	for i in range(15):
		# 		cv2.line(image,tuple(coords_temp[i]),tuple(coords_temp[i+1]),[255,0,255],thickness=3)
		# else:
		# 	xp = np.linspace(0, 400, 30)
		# 	curve_poly=np.poly1d(curve_1)
		# 	yp=curve_poly(xp)
		# 	curve_coords=np.squeeze(np.array([yp,xp],dtype=np.int32)).T
		# 	for i in range(len(xp)-1):
		# 		cv2.line(image,tuple(curve_coords[i]),tuple(curve_coords[i+1]),[255,0,255],thickness=3)
		# 	curve_poly=np.poly1d(curve_2)
		# 	yp=curve_poly(xp)
		# 	curve_coords=np.squeeze(np.array([xp,yp],dtype=np.int32)).T
		# 	for i in range(len(xp)-1):
		# 		cv2.line(image,tuple(curve_coords[i]),tuple(curve_coords[i+1]),[0,0,255],thickness=3)
		# 	curve_poly=np.poly1d(curve_3)
		# 	yp=curve_poly(xp)
		# 	curve_coords=np.squeeze(np.array([xp,yp],dtype=np.int32)).T
		# 	for i in range(len(xp)-1):
		# 		cv2.line(image,tuple(curve_coords[i]),tuple(curve_coords[i+1]),[255,0,0],thickness=3)
		# 	curve_poly=np.poly1d(curve_4)
		# 	yp=curve_poly(xp)
		# 	curve_coords=np.squeeze(np.array([xp,yp],dtype=np.int32)).T
		# 	for i in range(len(xp)-1):
		# 		cv2.line(image,tuple(curve_coords[i]),tuple(curve_coords[i+1]),[0,255,255],thickness=3)
		# out_CRF = self.postprocessor(np.array(img),np.asarray(out_))
		



		# print "PROCESSED CRF:",out_CRF.shape
		# out_=np.asarray(out_CRF)
		# print out_.shape
		# out11 = out_[0,:,0].reshape(self.batch_size,self.img_h,self.img_w),
		# out22 = out_[0,:,1].reshape(self.batch_size,self.img_h,self.img_w)
		# out33 = out_[0,:,2].reshape(self.batch_size,self.img_h,self.img_w)
		# out44 = out_[0,:,3].reshape(self.batch_size,self.img_h,self.img_w)
		# out55 = out_[0,:,4].reshape(self.batch_size,self.img_h,self.img_w)
		# # out11 = out_[0]
		# # out22 = out_[1]#,:,1].reshape(batch_size,img_h,img_w)
		# # out33 = out_[2]#,:,2].reshape(batch_size,img_h,img_w)
		# # out44 = out_[3]#,:,3].reshape(batch_size,img_h,img_w)
		# # out55 = out_[4]
		
		# threshold = 0.8
		# out_pro = np.array((out11[0]>threshold)*255,dtype=np.uint8).reshape(self.img_h,self.img_w)
		# out_pro1 = np.array((out22[0]>threshold)*255,dtype=np.uint8).reshape(self.img_h,self.img_w)
		# out_pro2 = np.array((out33[0]>threshold)*255,dtype=np.uint8).reshape(self.img_h,self.img_w)
		# out_pro3 = np.array((out44[0]>threshold)*255,dtype=np.uint8).reshape(self.img_h,self.img_w)
		# out_pro4 = np.array((out55[0]>threshold)*255,dtype=np.uint8).reshape(self.img_h,self.img_w)
		# # print out_pro.shape
		# out_pro = np.stack((out_pro,)*3, axis=-1)*self.lane_color_1
		# out_pro1 = np.stack((out_pro1,)*3, axis=-1)*self.lane_color_2
		# out_pro2 = np.stack((out_pro2,)*3, axis=-1)*self.lane_color_3
		# out_pro3 = np.stack((out_pro3,)*3, axis=-1)*self.lane_color_4
		# out_pro4 = np.stack((out_pro4,)*3, axis=-1)*self.lane_color_5
		# # print out_pro.shape
		# out_image = np.array(img)
		# # print out_image.shape
		# # print out_pro.shape
		# out_image = cv2.addWeighted(out_image,1.0,out_pro,0.4,0)
		# out_image = cv2.addWeighted(out_image,1.0,out_pro1,0.4,0)
		# out_image = cv2.addWeighted(out_image,1.0,out_pro2,0.4,0)
		# out_image = cv2.addWeighted(out_image,1.0,out_pro3,0.5,0)
		# out_image = cv2.addWeighted(out_image,1.0,out_pro4,0.5,0)
		# out_image = out_image[:,:,[2,1,0]]
		# cv2.imshow("dd:-",out_image)

		# out_=np.asarray(out_)
		# print out_.shape
		# out11 = out_[0,:,:,:,0].reshape(self.batch_size,img_h,img_w)
		# out22 = out_[0,:,:,:,1].reshape(self.batch_size,img_h,img_w)
		# out33 = out_[0,:,:,:,2].reshape(self.batch_size,img_h,img_w)
		# out44 = out_[0,:,:,:,3].reshape(self.batch_size,img_h,img_w)
		# out55 = out_[0,:,:,:,4].reshape(self.batch_size,img_h,img_w)
		# threshold = 0.8
		# out_pro = np.array((out11[0]>threshold)*255,dtype=np.uint8)
		# out_pro1 = np.array((out22[0]>threshold)*255,dtype=np.uint8)
		# out_pro2 = np.array((out33[0]>threshold)*255,dtype=np.uint8)
		# out_pro3 = np.array((out44[0]>threshold)*255,dtype=np.uint8)
		# out_pro4 = np.array((out55[0]>threshold)*255,dtype=np.uint8)
		# # print out_pro.shape
		# out_pro = np.stack((out_pro,)*3, axis=-1)*self.lane_color_1
		# out_pro1 = np.stack((out_pro1,)*3, axis=-1)*self.lane_color_2
		# out_pro2 = np.stack((out_pro2,)*3, axis=-1)*self.lane_color_3
		# out_pro3 = np.stack((out_pro3,)*3, axis=-1)*self.lane_color_4
		# out_pro4 = np.stack((out_pro4,)*3, axis=-1)*self.lane_color_5
		# # print out_pro.shape
		# out_image = np.array(img)
		# print out_image.shape
		# print out_pro.shape
		# out_image = cv2.addWeighted(out_image,1.0,out_pro,0.3,0)
		# out_image = cv2.addWeighted(out_image,1.0,out_pro1,0.3,0)
		# out_image = cv2.addWeighted(out_image,1.0,out_pro2,0.3,0)
		# out_image = cv2.addWeighted(out_image,1.0,out_pro3,0.5,0)
		# out_image = cv2.addWeighted(out_image,1.0,out_pro4,0.5,0)
		# # cv2.imshow("d",out_pro)
		# # cv2.imshow("d1",out_pro1)
		# # cv2.imshow("d2",out_pro2)
		# # cv2.imshow("d3",out_pro3)
		# # cv2.imshow("d4",out_pro4)
		# out_image = out_image[:,:,[2,1,0]]
		# cv2.imshow("CRF:-",out_image)
		# # cv2.imshow("CurveF:-",image)
		# cv2.waitKey(1)
		return out_image
		# key = cv2.waitKey(20)
		# logging.info("Weights loaded successfully.")

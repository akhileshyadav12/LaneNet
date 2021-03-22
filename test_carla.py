import tensorflow as tf 
import numpy as np 
# import matplotlib.pyplot as plt 
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
				self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()
				gpu_options = tf.GPUOptions(allow_growth=True)

				# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
				# init session
				self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
		
				self.saver = tf.train.Saver()
				self.saver.restore(self.sess, "/home/docker_share/ranjith_dev/lanenet/RUNS/model_400.ckpt")
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


		for i in range(1):
			prev = time.time()

			bin_seg,ins_seg= self.sess.run([self.output_logits['binary_seg_prob'],self.output_logits['instance_seg_logits']], feed_dict=feed)
			now_=time.time()

			print (now_-prev)
		bin_seg=np.array(bin_seg)[0]
		ins_seg=np.array(ins_seg)[0]
		# print ins_seg.shape
		out11 = bin_seg[:,:,0]#.reshape(batch_size,img_h,img_w)
		out22 = bin_seg[:,:,1]#.reshape(batch_size,img_h,img_w)
		# out33 = out_[:,:,:,2].reshape(batch_size,img_h,img_w)
		# out44 = out_[:,:,:,3].reshape(batch_size,img_h,img_w)
		# out55 = out_[:,:,:,4].reshape(batch_size,img_h,img_w)
		lane_color_1=[1,1,0]
		out_pro = np.array((out11>0.7)*255,dtype=np.uint8)
		out_pro1 = np.array((out22>0.7)*255,dtype=np.uint8)

		out_pro = np.stack((out_pro,)*3, axis=-1)*lane_color_1
		out_image = np.array(img)
		out_image_ = cv2.addWeighted(out_image,1.0,out_pro,0.5,0,dtype=cv2.CV_32F)

		out_image_=np.array(out_image_,dtype=np.uint8)
		# cv2.imshow("d",out_image_)
		# cv2.waitKey(0)
		postprocess_result = self.postprocessor.postprocess(
			binary_seg_result=bin_seg[:,:,0],
			instance_seg_result=ins_seg,
			source_image=img
		)

		# print postprocess_result.shape
		file_='meta.tsv'
		file_open=open(file_,'w')
		for i in range(postprocess_result.shape[0]):
			feat = postprocess_result[0,:]
			string=''
			for ii in range(postprocess_result.shape[1]):
				string+=str(postprocess_result[i][ii])+"\t"
			string=string[:-2]+"\n"
			file_open.write(string)

		# mask_image = postprocess_result['mask_image']
		return ins_seg

import tensorflow as tf 
import numpy as np 
import argparse
import cv2
import time
from tensorflow.python.platform import gfile
from post_process_lane_1 import predict_postprocess


# GRAPH_PB_PATH = 'RUNS/model_495_v1_0_new.pb'
GRAPH_PB_PATH = 'frozen_models/trt_model_101.pb'

class BClone(object):
	"""docstring for ClassName"""
	def __init__(self):
		self.image=[]

	def model_initialize_func(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.2
		self.sess=tf.Session(config=config)
		with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
			graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		self.sess.graph.as_default()
		tf.import_graph_def(graph_def, name='')
		# graph_nodes=[n for n in graph_def.node]
		# names = []
		# for t in graph_nodes:
		# 	names.append(t.name)
		# print(names)
		graph = tf.get_default_graph()
		output_node_names=['Input/X:0','Binary_Seg/FULL_CONV_binary:0','Instance_seg/FULL_CONV_instance:0']
		self.inp_img = graph.get_tensor_by_name(output_node_names[0])
		self.out_val = graph.get_tensor_by_name(output_node_names[1])
		self.out_val2 = graph.get_tensor_by_name(output_node_names[2])
		
		self.model_initialize=True
		
	def predict_func(self,img):

		for i in range(1):
			prev=time.time()
			steer_val,steer_val1=self.sess.run([self.out_val,self.out_val2],feed_dict={self.inp_img:[img]})
			predict_postprocess(img,steer_val,steer_val1)

			now=time.time()
			print (now-prev)
			print("______________________")
			print(steer_val.shape, steer_val1.shape)
		return steer_val,steer_val1

	def img_callback(self):

		print ("image received")
		try:
			img= cv2.imread("data/training_data_example/image/0000.png")
			img=cv2.resize(img,(640,480))
			# self.image = np.add(np.divide(cv_image,127.5),-1.0)

			valu,valu2=self.predict_func(img)
			cv2.imwrite("pred1.png",valu[0][:,:,1])
			cv2.imwrite("pred21.png",valu2[0][:,:,0]*255)
			cv2.imshow("Dfa",valu[0][:,:,1])
			cv2.imshow("dfa",valu2[0][:,:,0]*255)
			cv2.waitKey()
			print ("Steer:",valu[0])
			print(valu[0][:,:,0].shape)
		except Exception as e:
			print (e)
		

if __name__ == '__main__':	
	clone_obj=BClone()
	clone_obj.model_initialize_func()
	clone_obj.img_callback()
	# rospy publish steer_val
	# rospy subscribe image 
	# rospy subscribe trigger







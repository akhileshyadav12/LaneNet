import tensorflow as tf 
import numpy as np 
import argparse
import cv2
import time
from tensorflow.python.platform import gfile
import post_process_lane
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

GRAPH_PB_PATH = 'RUNS/model_300_v1_0_new_lane.pb'

class BClone(object):
	"""docstring for ClassName"""
	def __init__(self):
		self.image=[]
		self.bridge = CvBridge()
		rospy.Subscriber("/camera_data", Image, self.img_callback,queue_size=1)

	# def model_initialize_func(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.15
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
		output_node_names=['Input/X:0','Binary_Seg/logits_to_softmax:0','Instance_seg/FULL_CONV_instance:0']
		self.inp_img = graph.get_tensor_by_name(output_node_names[0])
		self.out_val = graph.get_tensor_by_name(output_node_names[1])
		self.out_val2 = graph.get_tensor_by_name(output_node_names[2])
		
		self.model_initialize=True
		
	def predict_func(self,img):

		# for i in range(1):
		prev=time.time()
		steer_val,steer_val1=self.sess.run([self.out_val,self.out_val2],feed_dict={self.inp_img:[img]})
		# steer_val=self.sess.run(tf.nn.softmax(steer_val))
		# print(steer_val)
		post_process_lane.predict_postprocess(img,steer_val,steer_val1)
		now=time.time()
		print (now-prev)
		return steer_val

	def img_callback(self,data):

		# # print ("image received")
		# # try:
		# # 	img= cv2.imread("62008.png")
		# # 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# # 	img=cv2.resize(img,(640,480))
		# # 	# self.image = np.add(np.divide(cv_image,127.5),-1.0)

			# valu=self.predict_func(img)
		# # 	# print ("Steer:",valu)
		# # except Exception as e:
		# # 	print (e)

		cv_image = self.bridge.imgmsg_to_cv2(data,"rgb8")
		# cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
		
		cv_image=cv2.resize(cv_image,((640,480)))
		valu=self.predict_func(cv_image)
		

if __name__ == '__main__':	
	rospy.init_node("Lane_computation")
	clone_obj=BClone()
	
	# clone_obj.model_initialize_func()
	rospy.spin()
	# clone_obj.img_callback()
	# rospy publish steer_val
	# rospy subscribe image 
	# rospy subscribe trigger







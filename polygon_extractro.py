import cv2
import numpy as  np 
import rospy
from sensor_msgs import point_cloud2
from std_msgs.msg import Bool,Float32
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sys
import json
import geometry_msgs.msg 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time 
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

# class poly_to_3d():

class poly_to_3d(object):
	def __init__(self):

		


		self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
							PointField('y', 4, PointField.FLOAT32, 1),
							PointField('z', 8, PointField.FLOAT32, 1),
							]

		self.header = Header()
		self.header.frame_id = "test"
		self.pub = rospy.Publisher("lane_img_cloud2", PointCloud2, queue_size=1)

		# x_resize_factor=
		rospy.Subscriber("/contour_data", numpy_msg(Floats), self.cnt_callback)

		camera_matrix=np.array([[ 9.50699875e+02, -9.26190231e+02, -7.23870766e+00,
				-1.45014909e+02],
			[ 4.95089881e+02, -2.83409600e+00, -9.24639768e+02,
				-8.03121944e+01],
			[ 9.99935210e-01, -1.00662336e-02, -5.31413872e-03,
				-1.86572656e-01]]
		)
		camera_matrix=np.array([[ 9.50699875e+02, -9.26190231e+02, -7.23870766e+00,
				-1.45014909e+02],
			[ 4.95089881e+02, -2.83409600e+00, -9.24639768e+02,
				-8.03121944e+01],
			[ 9.99935210e-01, -1.00662336e-02, -5.31413872e-03,
				-1.86572656e-01]]
		)

		intrinsic_matrix=np.array( [[916.5757063, 0. ,1920/2.],
		[  0.   ,      921.99825142, 1000/2. ],
		[  0.    ,       0.    ,       1.        ]])



		z_remove=np.eye(4,3)
		z_remove[0,2] = 0.0
		z_remove[1,2] = 0.0
		z_remove[2,2]=-1.65
		z_remove[3][2]=1

		H_mat = np.matmul(camera_matrix,z_remove)
		self.H_inv=np.linalg.inv(H_mat)
		
	def cnt_callback(self,pts):
		contours=np.array(pts.data)
		contours=np.reshape(contours,(int(contours.shape[0]/2),2))
		print(contours[:10])
		print(contours.shape)
		# img= cv2.imread("6200_mask_1.png")

		# mask = np.array(np.all(img == [255,0,0],axis=2),dtype=np.uint8)*255
		# contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		# img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
		# contours=np.squeeze(np.array(contours[0]))
		# print(contours[:10])
		# contours = np.roll(contou rs,1,axis=1)

		# contours=np.append(contours,np.ones((contours.shape[0],1)),axis=1)
		# print(contours.shape)
		# cv2.imshow("d",mask)
		# cv2.imshow("d1",img)
		# cv2.waitKey(0)

		contours=np.append(contours,np.ones((contours.shape[0],1)),axis=1)

		pts=np.matmul(self.H_inv,contours.T).T
		tmp=[]
		for pt in pts:
			pt_tmp=pt/float(pt[2])
			pt_tmp[2]=-1.64
			tmp.append(pt_tmp)

		tmp=np.array(tmp)
		print(tmp)
		# tmp[]
		print(tmp.shape)
		
		pc2 = point_cloud2.create_cloud(self.header, self.fields, tmp)
		pc2.header.stamp = rospy.Time.now()
		self.pub.publish(pc2)
		# while not rospy.is_shutdown():
		# 	# print("Dsfs")
		# 	self.pub.publish(pc2)
		# 	rospy.sleep(1)

if __name__ == '__main__':
	rospy.init_node("Polygon_ext")	
	clone_obj=poly_to_3d()
	rospy.spin()
	# rospy publish steer_val
	# rospy subscribe image 
	# rospy subscribe trigger

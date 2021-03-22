import os
import cv2
import numpy as np
# import test_carla
# img = cv2.imread("6936.png")


list_=os.listdir("data/instance")

for i in list_:
	ins_img = cv2.imread("data/instance/"+i)


	label=np.unique(ins_img)


	out_value = np.zeros((270,480),dtype=np.uint8)
	for i in range(len(label)-1):
		zero=np.zeros((ins_img.shape[0],ins_img.shape[1]),dtype=np.uint8)
		zero[np.all(ins_img==[(i+1),(i+1),(i+1)],axis=2)]=[255]

		res_zero=cv2.resize(zero,(480,270))
		res_zero[res_zero>0]=[(i+1)*50]

		out_value = np.bitwise_or(out_value,res_zero)


		cv2.imshow("d",zero)
		cv2.imshow("d1",out_value)
		cv2.waitKey(0)


	ins_img=out_value

	cv2.imshow("d",ins_img)
	cv2.waitKey(0)
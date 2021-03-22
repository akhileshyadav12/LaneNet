import tensorflow as tf 
import numpy as np 
# import matplotlib.pyplot as plt 
import math
import time
import argparse
# from PIL import Image as Image_pil
from ENet.enet import ENet
import cv2
# import lanenet_postprocess 
# from sklearn.decomposition import PCA
# from sklearn.cluster import DBSCAN

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge=CvBridge()
mask_pub = rospy.Publisher('/lane_mask', Image,queue_size=1)

print (time.time())

# class model_predict(object):
# 	def __init__(self):
# 		self.pers_points()
# 		self.lane_color_1 =np.array([1,1,0],dtype=np.uint8)
# 		self.lane_color_2 =np.array([1,0,1],dtype=np.uint8)
# 		self.lane_color_3 =np.array([0,1,1],dtype=np.uint8)
# 		self.lane_color_4 =np.array([0,0,1],dtype=np.uint8)
# 		self.lane_color_5 =np.array([0,0,0],dtype=np.uint8)
img_h=  480#270
img_w = 640#480 #28x28
# 		self.n_channels = 3

# 		self.batch_size =1
	
# 		with tf.device('/device:GPU:0'):
# 			with tf.Graph().as_default():
# 				img_h=480
# 				n_channels=3
# 				img_w= 640
lane_color_1=[[255,255,0],[0,255,0],[255,0,0],[0,0,255],[255,0,255],[0,255,255]]
# 				self.image_pl = tf.placeholder(tf.float32,shape=[None,img_h,img_w,n_channels])
# pca = PCA(n_components=3)
# 				network= ENet()
# 				self.output_logits = network.model(self.image_pl,skip_connection=True,
# 					batch_size=self.batch_size,stage_two_repeat=2,
# 					is_training=False,num_features_instance=8,
# 					num_classes=2,scope="LaneNet")
# 				gpu_options = tf.GPUOptions(allow_growth=True)
#                 self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
		
# 				self.saver = tf.train.Saver()
# 				self.saver.restore(self.sess, "/home/docker_share/ranjith_dev/laneNet/RUNS/highway/model_500.ckpt")
# 		#return sess

# 	def pers_points(self):
src=np.float32([[122,266],
            [154,230],
            [340,230],
            [376,266]])


dst=np.float32([[122,1766],
            [122,1700],
            [376,1700],
            [376,1766]])
M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

# 	def destroy_winds(self):
# 		cv2.destroyAllWindows()
# 		print ("CV Windows Closed")
# 	def remove_sess(self):
# 		self.sess.close()
# 		cv2.destroyAllWindows()
# 		print ("Session Closexd::::::::::::::::::::::::::::::::::")

def lane_fit(lane_1,out_pro):
    z = np.polyfit(lane_1.T[1], lane_1.T[0], 1)
    # params.append(z)
    f = np.poly1d(z)
    t_l=range(250, 480)
    t_l_1 = list(map(int,f(t_l)))
    temp_list=np.array(list(map(list,zip(t_l_1,t_l))))
    # print(temp_list)

    # for i in range(len(temp_out_idx)-1):
    #     cv2.line(pers_samp_img,tuple([int(temp_out_idx[i][0]),int(temp_out_idx[i][1])]),
    #                             tuple([int(temp_out_idx[i+1][0]),int(temp_out_idx[i+1][1])]),color=[0,255,0],thickness=3)
    for i in range(len(temp_list)-1):
        cv2.line(out_pro,tuple([int(temp_list[i][0]),int(temp_list[i][1])]),
                                tuple([int(temp_list[i+1][0]),int(temp_list[i+1][1])]),color=[127],thickness=3)
    return out_pro,temp_list

def predict_postprocess(img,bin_seg,ins_seg):
    
    bin_seg=np.array(bin_seg)[0]
    # ins_seg=np.array(ins_seg)[0]
    # print ins_seg.shape
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    out11 = bin_seg[:,:,0]#.reshape(batch_size,img_h,img_w)
    
    out_pro = np.array((out11>0.8)*255,dtype=np.uint8)
    cv2.imshow("output",out_pro)
    cv2.waitKey(1)
    # out_pro1 = np.array((out22>0.7)*255,dtype=np.uint8)
    contours, hierarchy = cv2.findContours(out_pro,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    min_vals=[]
    contours=np.squeeze(np.array(contours))
    # print((contours))
    for cnt in contours:
        # print(cnt)
        tmp_min=100000
        if len(cnt)>100:
            for pt in cnt:
                if tmp_min>np.sqrt(np.sum(np.square(pt[0]-[360,480]))):
                    tmp_min=np.sqrt(np.sum(np.square(pt[0]-[360,480])))
        min_vals.append(tmp_min)
    print(min_vals)
    cnt_sort=np.argsort(min_vals)
    # out_pro = cv2.drawContours(out_pro, contours[cnt_sort[1]], -1, (127), 5)

    lane_1=np.squeeze(contours[cnt_sort[0]])
    lane_2=np.squeeze(contours[cnt_sort[1]])

    out_pro,l1 = lane_fit(lane_1,out_pro)
    out_pro,l2 = lane_fit(lane_2,out_pro)
    # print(np.concatenate((l1,np.flipud(l2))))
    cv2.fillPoly(out_pro,pts = [np.concatenate((l1,np.flipud(l2)))], color = (255))

    msg = bridge.cv2_to_imgmsg(out_pro,"mono8")
    mask_pub.publish(msg)
    
            # print(np.sqrt(np.sum(np.square(pt[0]-[360,480]))))
    # print(contours[3])
    # return
    # out_pro[::2,:]=0 # To make alternate pixels zero (sparse output)
    # out_pro[:,::2]=0

    now_=time.time()
    # print ("Inference : ",now_-prev)


    cv2.imshow("Ego_lane",out_pro)

    return
    # idx=np.where(out_pro>=250)
    # print(idx)
    # embed_features=ins_seg[idx]
    # print(embed_features.shape)


    # file_='meta/meta_'+"test"+'.tsv'
    # file_open=open(file_,'w')
    # # print (len(feat))
    
    # for ii in range(embed_features.shape[0]):
    #     feat_temp = embed_features[ii,:]
    #     string=''
    #     for iii in range(feat_temp.shape[0]):
    #         string+=str(feat_temp[iii])+"\t"
    #     string=string[:-2]+"\n"
    #     file_open.write(string)
    #     file_open.flush()
    # return


    # lane_coords = np.vstack((idx[1], idx[0])).transpose()
    
    # # print (embed_features.shape)
    # # prev = time.time()
    # embed_features=pca.fit_transform(embed_features) 
    # # now_=time.time()
    # # print ("PCA :",now_-prev)
    # # prev = time.time()
    # lab=DBSCAN(eps=1,  metric='euclidean',min_samples=100, metric_params=None, algorithm='kd_tree').fit(embed_features)
    # # now_=time.time()
    # # print ("DBSCAN :",now_-prev)
    # db_labels = lab.labels_
    # print(db_labels)
    # unique_labels = np.unique(db_labels)
    
    
    
    # out_pro = np.stack((out_pro,)*3, axis=-1)
    # print (unique_labels)
    
    # min_vals=[]
    # split_lanes=[]
    # ii_32 = np.iinfo(np.int32)
    # for i in range(len(unique_labels)):
    #     if unique_labels[i]==-1:
    #         continue
    #     temp_label=np.where(db_labels==unique_labels[i])
    #     # print (len(temp_label[0]))
    #     temp_idx=lane_coords[temp_label]
    #     split_lanes.append(temp_idx)
    #     dist=np.sum(np.square(((temp_idx[temp_idx.T[1]>(img_h-20)])-[img_w/2,img_h])),axis=1)
    #     if (len(dist)>1) and (len(temp_label[0])>150):
    #         min_vals.append(np.amin(np.sum(np.square(((temp_idx[temp_idx.T[1]>(img_h-20)])-[img_w/2,img_h])),axis=1)))
    #     else:
    #         min_vals.append(ii_32.max)
    #     # print (temp_idx[temp_idx.T[1]<30])
    #     # print (temp_idx[temp_idx.T[1]<30]-[self.img_h,self.img_w/2])
    #     # print (np.square(temp_idx[temp_idx.T[1]>(self.img_h-20)]-[self.img_h,self.img_w/2]))
    #     img[tuple(temp_idx.T[1]),tuple(temp_idx.T[0])]=lane_color_1[i]#*255
    #     # print (dist)
    #     # print (min_vals)
    #     # cv2.imshow("sa",out_pro)
    #     # cv2.waitKey(0)

    # lanes=np.argsort(min_vals)

    # np.savez("data.npz",lanes=lanes,split_lanes=split_lanes,M=M,M_inv=M_inv)
    # min_dist=min_vals[lanes[0]]
    # # print ("LANES:___",lanes)
    # params=[]
    # params_inv=[]
    # pers_samp_img=np.zeros((2400,800,3))
    # for count,i in enumerate(lanes):
    #     # print (count,i)
    #     if count>1:
    #         break
    #     temp_idx=split_lanes[i]
    #     # selected_=temp_idx[temp_idx.T[1]>(self.img_h-70)]
    #     temp_idx= (np.append(temp_idx,np.ones((temp_idx.shape[0],1)),axis=1))
    #     temp_out_idx= np.matmul(M,temp_idx.T)
    #     temp_out_idx=temp_out_idx/temp_out_idx[2]
    #     temp_out_idx=temp_out_idx.T
    #     z = np.polyfit(temp_out_idx.T[1], temp_out_idx.T[0], 2)
    #     params.append(z)
    #     f = np.poly1d(z)
    #     t_l=range(0, 1750)
    #     t_l_1 = f(t_l)
    #     temp_list=np.array(map(list,zip(t_l_1,t_l)))


    #     for i in range(len(temp_out_idx)-1):
    #         cv2.line(pers_samp_img,tuple([int(temp_out_idx[i][0]),int(temp_out_idx[i][1])]),
    #                                 tuple([int(temp_out_idx[i+1][0]),int(temp_out_idx[i+1][1])]),color=[0,255,0],thickness=3)
    #     for i in range(len(temp_list)-1):
    #         cv2.line(pers_samp_img,tuple([int(temp_list[i][0]),int(temp_list[i][1])]),
    #                                 tuple([int(temp_list[i+1][0]),int(temp_list[i+1][1])]),color=[255,255,0],thickness=3)
    #     temp_list= (np.append(temp_list,np.ones((temp_list.shape[0],1)),axis=1))
    #     temp_list= np.matmul(M_inv,temp_list.T)
    #     temp_list=temp_list/temp_list[2]
    #     temp_list=temp_list.T


    #     z = np.polyfit(temp_list.T[1][-200:], temp_list.T[0][-200:], 1)
    #     params_inv.append(z)
    #     # f = np.poly1d(z)
    #     # for i_line in range(self.img_h-150, self.img_h):
    #     # 	# print (i_line,f(i_line))
    #     # 	cv2.line(out_pro,tuple([int(f(i_line)),int(i_line)]),tuple([int(f(i_line+1)),int(i_line+1)]),color=[255,255,0],thickness=3)
    #     # print ("dsfsfsf:---",temp_list)
    #     for i_line in temp_list[-200:]:
    #         # print (i_line,f(i_line))
    #         cv2.line(img,tuple([int(i_line[0]),int(i_line[1])]),tuple([int(i_line[0]),int(i_line[1])]),color=[255,255,0],thickness=3)


    # z=(params[0]+params[1])/float(2)
    # f = np.poly1d(z)
    # t_l=np.linspace(0,1750,num=10)
    # t_l_1 = f(t_l)
    # temp_list=np.array(map(list,zip(t_l_1,t_l)))

    # # dy = np.gradient(t_l_1)
    # # dy_2  = np.gradient(dy)
    # # curvature=((1+(dy)**2)**(3/2))/abs(dy_2)

    # # dx=np.gradient(temp_list[:,0])
    # # dy=np.gradient(temp_list[:,1])

    # # dx_2=np.gradient(dx)
    # # dy_2=np.gradient(dy)
    # # print (dx_2*dy)
    # # print (dx_2*dy-dx*dy_2)
    # # print (dx**2)
    # # print (float((dx**2+dy**2)**(1.5)))

    # # curvature=(dx_2*dy-dx*dy_2)/((dx**2+dy**2)**(1.5))


    # for i in range(len(temp_list)-1):
    #     cv2.line(pers_samp_img,tuple([int(temp_list[i][0]),int(temp_list[i][1])]),
    #                             tuple([int(temp_list[i+1][0]),int(temp_list[i+1][1])]),color=[255,0,255],thickness=3)
    # temp_list= (np.append(temp_list,np.ones((temp_list.shape[0],1)),axis=1))
    # temp_list= np.matmul(M_inv,temp_list.T)
    # temp_list=temp_list/temp_list[2]
    # temp_list=temp_list.T

    # for i_line in temp_list:
    #     # print (i_line,f(i_line))
    #     cv2.line(img,tuple([int(i_line[0]),int(i_line[1])]),tuple([int(i_line[0]),int(i_line[1])]),color=[255,0,255],thickness=3)
    # cv2.imshow("big",pers_samp_img)

    # print ("---------------Int:------------------")

    # # print ("Curvature: ", curvature)
    # # print (params)
    # inter_x=int(float(params_inv[1][1]-params_inv[0][1])/float(params_inv[0][0]-params_inv[1][0]))
    # inter_y=int(float(params_inv[1][0]*inter_x)+params_inv[1][1])
    # print ("intersection:",inter_x,inter_y)
    # slope_steer=float((inter_y+1)-(img_w/2))/float(inter_x-img_h)
    # angle=-(90-(math.atan(float(-1/slope_steer))*180/math.pi))#*math.pi/180
    # # print (angle)
    # if(angle<-100):
    #         angle = angle+180
    # print ("Angle:",angle)
    # angle=-angle/float(60)
    # print ("-----------------")
    # cv2.line(img,tuple([int(inter_y),int(inter_x)]),tuple([int(inter_y+1),int(inter_x+1)]),color=[255,255,255],thickness=5)
    # # out_pro = np.stack((out_pro,)*3, axis=-1)*lane_color_1
    # # out_image = np.array(img)
    # # out_image_ = cv2.addWeighted(out_image,1.0,out_pro,0.3,0,dtype=cv2.CV_32F)

    # # out_image_=np.array(out_image_,dtype=np.uint8)
    # # cv2.imshow("img",img)
    # # dst_pres = cv2.warpPerspective(out_pro,self.M,(800,1800),flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
    # cv2.imshow("output",img)
    # # cv2.imshow("pers",dst_pres)
    # cv2.waitKey(0)
    
    # # lane_color_1=[1,0,1]
    # # out_pro = np.array((out11>0.7)*255,dtype=np.uint8)
    # # out_pro1 = np.array((out22>0.7)*255,dtype=np.uint8)

    # # out_pro = np.stack((out_pro,)*3, axis=-1)*lane_color_1
    # # out_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    # # out_image_ = cv2.addWeighted(out_image,1.0,out_pro,0.5,0,dtype=cv2.CV_32F)
    # # print ("After Weight")
    # # out_image_=np.array(out_image_,dtype=np.uint8)
    
    # # # cv2.imwrite("mask_latest.png",out_image_)
    # # cv2.imshow("mask_latest",out_image_)
    # # cv2.waitKey(1)
    # # print ("After Display")
    # # postprocess_result = self.postprocessor.postprocess(
    # # 	binary_seg_result=bin_seg[:,:,0],
    # # 	instance_seg_result=ins_seg,
    # # 	source_image=img
    # # )

    # # print postprocess_result.shape
    # # file_='meta.tsv'
    # # file_open=open(file_,'w')
    # # for i in range(postprocess_result.shape[0]):
    # # 	feat = postprocess_result[0,:]
    # # 	string=''
    # # 	for ii in range(postprocess_result.shape[1]):
    # # 		string+=str(postprocess_result[i][ii])+"\t"
    # # 	string=string[:-2]+"\n"
    # # 	file_open.write(string)

    # # mask_image = postprocess_result['mask_image']
    # angle=0
    # return angle,min_dist

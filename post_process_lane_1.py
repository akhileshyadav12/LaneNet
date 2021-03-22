import numpy as np 
import cv2
import time

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
    # cv2.imshow("output",out_pro)
    # cv2.waitKey(1)
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
    out_pro = cv2.drawContours(out_pro, contours[cnt_sort[1]], -1, (127), 5)

    lane_1=np.squeeze(contours[cnt_sort[0]])
    lane_2=np.squeeze(contours[cnt_sort[1]])

    out_pro,l1 = lane_fit(lane_1,out_pro)
    out_pro,l2 = lane_fit(lane_2,out_pro)
    # print(np.concatenate((l1,np.flipud(l2))))
    cv2.fillPoly(out_pro,pts = [np.concatenate((l1,np.flipud(l2)))], color = (255))

    # msg = bridge.cv2_to_imgmsg(out_pro,"mono8")
    # mask_pub.publish(msg)
    
            # print(np.sqrt(np.sum(np.square(pt[0]-[360,480]))))
    # print(contours[3])
    # return
    # out_pro[::2,:]=0 # To make alternate pixels zero (sparse output)
    # out_pro[:,::2]=0

    now_=time.time()
    # print ("Inference : ",now_-prev)


    # cv2.imshow("Ego_lane",out_pro)
    cv2.imwrite("ego_lane.png",out_pro)
    return
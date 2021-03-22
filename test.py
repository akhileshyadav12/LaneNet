import os
import cv2
import numpy as np
import test_carla
# img = cv2.imread("6936.png")

img = cv2.imread("data/center/160.png")
ins_img = cv2.imread("data/instance/160.png")


label=np.unique(ins_img)



out_value = np.zeros((270,480),dtype=np.uint8)
for i in range(len(label)-1):
	zero=np.zeros((ins_img.shape[0],ins_img.shape[1]),dtype=np.uint8)
	zero[np.all(ins_img==[(i+1),(i+1),(i+1)],axis=2)]=[255]

	res_zero=cv2.resize(zero,(480,270))
	res_zero[res_zero>0]=[(i+1)*50]

	out_value = np.bitwise_or(out_value,res_zero)


	# cv2.imshow("d",zero)
	# cv2.imshow("d1",out_value)
	# cv2.waitKey(0)


ins_img=out_value
# print (zero.shape)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(480,270))
# ins_img = cv2.resize(ins_img,(480,270))


# ins_img=ins_img*50
# cv2.imshow("d",ins_img)


# cv2.waitKey(0)

model=test_carla.model_predict()

ins_seg = model.predict(img)

# ins_img=np.reshape(ins_im)


label=np.unique(ins_img)
print(label)
feat=[]
for i in range(len(label)-1):
	select= ins_img==[(i+1)*50]

	feat.append(ins_seg[select])

file_='meta.tsv'
file_open=open(file_,'w')
print(len(feat))
for i in range(len(feat)):
	arr= np.array(feat[i])
	print (arr.shape)
	for ii in range(arr.shape[0]):
		feat_temp = arr[ii,:]
		string=''
		for iii in range(feat_temp.shape[0]):
			string+=str(feat_temp[iii])+"\t"
		string=string[:-2]+"\n"
		file_open.write(string)


file_1="meta_1.tsv"
file_open_1=open(file_1,'w')

for i in range(len(feat)):
	arr = np.array(feat[i]).shape[0]
	string=''
	for ii in range(arr):
		string=str(i)+"\n"
		file_open_1.write(string)
# print "Main_file"
# print feat_1.shape


# for i in range(postprocess_result.shape[0]):
# 	feat = postprocess_result[0,:]
# 	string=''
# 	for ii in range(postprocess_result.shape[1]):
# 		string+=str(postprocess_result[i][ii])+"\t"
# 	string=string[:-2]+"\n"
# 	file_open.write(string)
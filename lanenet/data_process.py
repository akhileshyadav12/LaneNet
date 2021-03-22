import tensorflow as tf 
import numpy as np 
import numpy.core.defchararray as np_f
import matplotlib.pyplot as plt 
import os
# Importing Dataset
import pandas as pd
# import read_csv
import json

def read_pandas_csv(data_dir,col_name,train_names,test_names):

	# print "FUNC--------------------"
	data_df = pd.read_csv(os.path.join(os.getcwd(), data_dir), usecols=col_name)
	# print data_df[['center','left','right']].values
	#yay dataframes, we can select rows and columns by their names
	#we'll store the camera images as our input data
	X = data_df[train_names].values
	#and our steering commands as our output data
	y = map(float,data_df[test_names].values)

	return np.array(X),np.array(y)



def Curve_extend(curve):
	poi = curve[0]
	poi2 = curve[5]
	x1,y1=poi2-poi
	m=y1/float(x1)
	x2 =poi[0]-(poi[1]-0)/float(m)
	curve=curve.tolist()
	curve.append([int(x2),0])
	pts=np.array(curve)
	sorted_=np.array(np.argsort(np.array(pts).T[1]),dtype=int)
	pts=pts[sorted_]

	return pts


def read_pandas_txt(data_dir,train=True):

	data_df = pd.read_csv(data_dir, header=None)#os.path.join(os.getcwd(), 
	temp_ = np.asarray([str(x[0]).split(" ") for x in data_df.values])
	bin_temp_=np_f.replace(temp_,['semantic'],['binary'])
	ins_temp_=np_f.replace(bin_temp_,['binary'],['instance'])
	print (bin_temp_,ins_temp_)
	print ("TOTAL NUMBERS:",temp_.shape)
	if train:
		X=temp_[:,0]
		y=bin_temp_[:,1]
		y_ins=ins_temp_[:,1]
		return np.array(X),np.array(y),np.array(y_ins)#,coords
	else:
		X=temp_[:,0]
		return np.array(X)



def validation_split(ratio,x_t,y_t,y_t_ins,coords_t):
	x_train_t,y_train_t,coords_t = x_t[:int(ratio*len(x_t))],y_t[:int(ratio*len(y_t))],coords_t[:int(ratio*len(coords_t))]


	x_valid,y_valid,coords_valid = x_t[int(ratio*len(x_t)):],y_t[int(ratio*len(y_t)):],coords_t[int(ratio*len(coords_t)):]

	return x_train_t,y_train_t,coords_t,x_valid,y_valid,coords_valid

def reformat(x,y):
	img_size,num_ch,num_class = int((x.shape[-1])),1,10
	# print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^::::,",x.shape[-1],x.shape[0],x.shape
	# print("IMAGD",img_size,num_ch,num_class)
	dataset = x.reshape((-1,img_size,img_size,num_ch)).astype(np.float32)
	labels = (np.arange(num_class)==y[:,None]).astype(np.float32)
	return dataset,labels

def load_data(mode='train'):
	mnist = tf.keras.datasets.mnist       
	(x_train,y_train),(x_test,y_test)=mnist.load_data()
	# print x_train,y_train
	if(mode=='train'):
		x_train1,y_train1 = reformat(x_train,y_train)
		x_train_t,y_train_t = x_train1[:int(0.80*len(x_train1))],y_train1[:int(0.80*len(y_train1))]
		x_valid,y_valid = x_train1[int(0.80*len(x_train1)):],y_train1[int(0.80*len(y_train1)):]
			
		return x_train_t,y_train_t,x_valid,y_valid
	elif mode == "test":                
		x_test,y_test = reformat(x_test,y_test)
				
	return x_test,y_test

def randomize(x,y,y_ins,is_list):
	permutation = np.random.permutation(y.shape[0])
	if is_list:
		shuffled_x = x[permutation]
	else:
		shuffled_x = x[permutation,:,:,:]

	# shuffled_coords=coords[permutation]
	shuffled_y = y[permutation]
	shuffled_y_ins = y_ins[permutation]
	return shuffled_x,shuffled_y,shuffled_y_ins #,shuffled_coords

def get_next_batch(x,y,start,end):
	x_batch = x[start:end]
	y_batch = y[start:end]
	return x_batch,y_batch


def plot_images(images,cls_true,cls_pred=None,title=None):
	fig, axes = plt.subplots(3,3,figsize=(9,9))
	fig.subplots_adjust(hspace=0.3,wspace=0.3)
	for i,ax in enumerate(axes.flat):
		ax.imshow(np.squeeze(images[i]),cmap="binary")
		if cls_pred is None:
			ax_title = "True: {0}".format(cls_true[i])
		else:
			ax_title = "True: {0}, Pred: {1}".format(cls_true[i],cls_pred[i])
		ax.set_title(ax_title)

		ax.set_xticks([])
		ax.set_yticks([])

	if title:
		plt.suptitle(title,size=20)
	plt.show(block=False)

def plot_example_errors(images,cls_true,cls_pred,title=None):
	incorrect = np.logical_not(np.equal(cls_pred,cls_true))
	incorrect_images = images[incorrect]
	cls_pred=cls_pred[incorrect]
	cls_true = cls_true[incorrect]
	plot_images(images=incorrect_images[0:9],
				cls_true = cls_true[0:9],
				cls_pred = cls_pred[0:9],
				title=title)
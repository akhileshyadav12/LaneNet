import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import os
# Importing Dataset
from tensorflow.examples.tutorials.mnist import input_data
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


def read_pandas_txt(data_dir,json_dir,train=True):

	# print "FUNC--------------------"
	data_df = pd.read_csv(os.path.join(os.getcwd(), data_dir), header=None)
	# x = dp.read_pandas_txt(args.data_dir,data_extract_col_name,train_col_names,test_col_names)
	#print "---------",data_df
	# print data_df[['center','left','right']].values
	#yay dataframes, we can select rows and columns by their names
	#we'll store the camera images as our input data
	file_=open(json_dir,'r')
	all_vals={}
	lines_=file_.readlines()


	coords_extract=False
	for i in lines_:
		# print "--------------"
		# print i
		k=json.loads(i)
		
		ll=np.array(k['ll'])
		
		ll=np.array(np.array(ll),dtype=np.float32)
		ll=Curve_extend(ll)
		# ll=np.array(np.array(ll)/np.array([1280.,720.]),dtype=np.float32)
		# print("#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",ll)
		l=np.array(k['l'])
		l=np.array(np.array(l),dtype=np.float32)
		l=Curve_extend(l)
		# l=np.array(np.array(l)/np.array([1280.,720.]),dtype=np.float32)
		
		rr=np.array(k['rr'])
		rr=np.array(np.array(rr),dtype=np.float32)
		rr=Curve_extend(rr)
		# rr=np.array(np.array(rr)/np.array([1280.,720.]),dtype=np.float32)
		
		r=np.array(k['r'])
		r=np.array(np.array(r),dtype=np.float32)
		r=Curve_extend(r)
		# r=np.array(np.array(r)/np.array([1280.,720.]),dtype=np.float32)
		if coords_extract:
			ll = np.array(ll).reshape((ll.shape[0]*ll.shape[1],1)).squeeze()
			l = np.array(l).reshape((l.shape[0]*l.shape[1],1)).squeeze()
			rr = np.array(rr).reshape((rr.shape[0]*rr.shape[1],1)).squeeze()
			r = np.array(r).reshape((r.shape[0]*r.shape[1],1)).squeeze()
		else:
			x,y=ll.T
			ll=np.array(np.polyfit(x,y,4)).squeeze()
			x,y=l.T
			l=np.array(np.polyfit(x,y,4)).squeeze()
			x,y=rr.T
			rr=np.array(np.polyfit(x,y,4)).squeeze()
			x,y=r.T
			r=np.array(np.polyfit(x,y,4)).squeeze()
			# print poly_
		dict_={k["name"]:[ll.tolist(),l.tolist(),r.tolist(),rr.tolist()]}
		# print dict_
		all_vals.update(dict_)

	temp_ = np.asarray([str(x[0]).split(" ") for x in data_df.values])
	print "TOTAL NUMBERS:",temp_.shape
	X=temp_[:,0]
	init=False
	keys=np.array(all_vals.keys())
	print "Keys:",keys.shape
	for i in X:
		sp=i.split("/")
		k=sp[-1][:-4]
		# all_vals[str(k)]

		if str(k) in keys:
			if not init:
				init=True
				coords=np.array([all_vals[str(k)]])
			else:
				coords=np.append(coords,[all_vals[str(k)]],axis=0)
			# print coords.shape
		else:
			element=np.squeeze(np.argwhere(temp_[:,0]==i))
			# print temp_[element]
			temp_= np.delete(temp_,np.argwhere(temp_[:,0]==i),axis=0)
			# print temp_[element]

	print "DICT MAPPED NUMBERS:",coords.shape
	print "New Temp:",temp_.shape
	# y=temp_[:,1]
	if train:
		X=temp_[:,0]
		y=temp_[:,1]
		return np.array(X),np.array(y),coords
	else:
		X=temp_[:,0]
		return np.array(X)



def validation_split(ratio,x_t,y_t,coords_t):
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

def randomize(x,y,coords,is_list):
	permutation = np.random.permutation(y.shape[0])
	if is_list:
		shuffled_x = x[permutation]
	else:
		shuffled_x = x[permutation,:,:,:]

	shuffled_coords=coords[permutation]
	shuffled_y = y[permutation]
	return shuffled_x,shuffled_y,shuffled_coords

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
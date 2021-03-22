import tensorflow as tf 
import cv2
import numpy as np
# filename = "/home/docker_share/ranjith_dev/docker_link/Town4/center/18764.png"
# image_string = tf.read_file(filename)
# image_decoded = tf.image.decode_png(image_string,channels=3,dtype=tf.uint8)



# count=0
# while count<1000:
# 	count+=1
# 	max_delta = 0.1
# 	# bright_ness = tf.image.random_brightness(image_decoded,    max_delta=0.2,    seed=None)
# 	# hue = tf.image.random_hue(image_decoded,    max_delta=0.15,    seed=None)
# 	# contrast = tf.image.random_contrast(image_decoded,    0.6,1.0)

# 	road_color =tf.constant([255,0,0],dtype=tf.uint8)
# 	# road = tf.image.resize_images(tf.cast(tf.reduce_all(tf.equal(image_decoded,road_color),axis=2),dtype=tf.uint8),[224,224])
# 	# road.set_shape([100, 100])

# 	img_h = 270
# 	img_w=480
# 	lane_color_1 =tf.constant([255,0,0],dtype=tf.uint8) #[255,255,0]

# 	# lane_color_2 =tf.constant([255,0,255],dtype=tf.uint8)

# 	# lane_color_3 =tf.constant([0,255,255],dtype=tf.uint8)
# 	# lane_color_4 =tf.constant([0,0,255],dtype=tf.uint8)
# 	# bg_color =tf.constant([0,0,0],dtype=tf.uint8)
		
# 	#inp_file_name = tf.strings.join([img_dir,filename])
# 	image_string = tf.read_file(filename)
# 	image_decoded = tf.image.decode_png(image_string,channels=0,dtype=tf.uint8)


# 	image_resized = tf.cast(tf.image.resize_images(image_decoded, [img_h, img_w]),dtype=tf.uint8)

# 	# lab_file_name = tf.strings.join([img_dir,label])
# 	# image_string = tf.read_file(lab_file_name)
# 	# label_decoded = tf.image.decode_png(image_string,channels=0,dtype=tf.uint8)
# 	# label_resized = tf.image.resize_images(label_decoded, [img_h, img_w])
		
# 	# road_im1 = tf.cast(tf.reduce_all(tf.equal(image_decoded,lane_color_1),axis=2),dtype=tf.uint8)
# 	# road_im1 = tf.cast(tf.reduce_all(tf.equal(image_decoded,lane_color_1),axis=2),dtype=tf.uint8)

# 	#road_im1 = tf.cast(tf.image.resize_images(image_decoded, [img_h, img_w]),dtype=tf.uint8)
# 	# road_im2 = tf.cast(tf.reduce_all(tf.equal(tf.cast(tf.image.resize_images(label_decoded, [img_h, img_w]),dtype=tf.uint8),lane_color_2),axis=2),dtype=tf.uint8)
# 	# road_im3 = tf.cast(tf.reduce_all(tf.equal(tf.cast(tf.image.resize_images(label_decoded, [img_h, img_w]),dtype=tf.uint8),lane_color_3),axis=2),dtype=tf.uint8)
# 	# road_im4 = tf.cast(tf.reduce_all(tf.equal(tf.cast(tf.image.resize_images(label_decoded, [img_h, img_w]),dtype=tf.uint8),lane_color_4),axis=2),dtype=tf.uint8)
# 	# bg_im = tf.cast(tf.reduce_all(tf.not_equal(tf.cast(tf.image.resize_images(label_decoded, [img_h, img_w]),dtype=tf.uint8),lane_color_1),axis=2),dtype=tf.uint8)

# 	stacked = tf.to_float(tf.stack((image_resized,image_resized,image_resized),axis=2))
# 	sess = tf.Session()
# 	with sess.as_default():
# 		#print stacked.get_shape()
# 		img = np.array(image_decoded.eval())
# 		label = np.array(bright_ness.eval())
# 		#print label
# 		cv2.imshow("d",label[:,:,::-1])
# 		cv2.imshow("d1",img[:,:,::-1])
# 		cv2.waitKey(0)
# 		# cv2.waitKey(0)
# 		#print label

t = tf.range(0,5*5*3,1)
t = tf.reshape(t,[5,5,3])
paddings = tf.constant([[0, 0], [0, 0],[0,5]])

x=tf.pad(t, paddings, "CONSTANT") 



sess = tf.Session()
with sess.as_default():
	#print stacked.get_shape()
	re=t.eval()
	print re

	img = x.eval()
	print img
	print img.shape
	# img = np.array(image_decoded.eval())
	# label = np.array(bright_ness.eval())
	# #print label
	# cv2.imshow("d",label[:,:,::-1])
	# cv2.imshow("d1",img[:,:,::-1])
	# cv2.waitKey(0)
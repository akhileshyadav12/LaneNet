import cv2, os
import numpy as np
import numpy.core.defchararray as np_f
import matplotlib.image as mpimg
import tensorflow as tf
# from PIL import Image_p
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
	"""
	Load RGB images from a file
	"""
	#print image_file
	return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
	"""
	Crop the image (removing the sky at the top and the car front at the bottom)
	"""
	return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
	"""
	Resize the image to the input shape used by the network model
	"""
	return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
	"""
	Convert the image from RGB to YUV (This is what the NVIDIA model does)
	"""
	return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
	"""
	Combine all preprocess functions into one
	"""
	image = crop(image)
	image = resize(image)
	image = rgb2yuv(image)
	return image


def choose_image(data_dir, center, left, right, steering_angle):
	"""
	Randomly choose an image from the center, left or right, and adjust
	the steering angle.
	"""
	choice = np.random.choice(3)
	if choice == 0:
		return load_image(data_dir, left), steering_angle + 0.2
	elif choice == 1:
		return load_image(data_dir, right), steering_angle - 0.2
	return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
	"""
	Randomly flipt the image left <-> right, and adjust the steering angle.
	"""
	if np.random.rand() < 0.5:
		image = cv2.flip(image, 1)
		steering_angle = -steering_angle
	return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
	"""
	Randomly shift the image virtially and horizontally (translation).
	"""
	trans_x = range_x * (np.random.rand() - 0.5)
	trans_y = range_y * (np.random.rand() - 0.5)
	steering_angle += trans_x * 0.002
	trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
	height, width = image.shape[:2]
	image = cv2.warpAffine(image, trans_m, (width, height))
	return image, steering_angle


def random_shadow(image):
	"""
	Generates and adds random shadow
	"""
	# (x1, y1) and (x2, y2) forms a line
	# xm, ym gives all the locations of the image
	x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
	x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
	xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

	# mathematically speaking, we want to set 1 below the line and zero otherwise
	# Our coordinate is up side down.  So, the above the line: 
	# (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
	# as x2 == x1 causes zero-division problem, we'll write it in the below form:
	# (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
	mask = np.zeros_like(image[:, :, 1])
	mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

	# choose which side should have shadow and adjust saturation
	cond = mask == np.random.randint(2)
	s_ratio = np.random.uniform(low=0.2, high=0.5)

	# adjust Saturation in HLS(Hue, Light, Saturation)
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
	return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
	"""
	Randomly adjust brightness of the image.
	"""
	# HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
	hsv[:,:,2] =  hsv[:,:,2] * ratio
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
	"""
	Generate an augumented image and adjust steering angle.
	(The steering angle is associated with the center image)
	"""

	image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
	image, steering_angle = random_flip(image, steering_angle)
	image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
	# image = random_shadow(image)
	image = random_brightness(image)
	return image, steering_angle

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label_bin,label_ins,img_dir):

	print ("___________________________________")
	print (filename,label_bin)
	# img = Image_p.open(filename)
	# label = Image_p.open(label)

	img_h = 480
	img_w= 640
	lane_color_1 =tf.constant([255,0,255],dtype=tf.uint8) #[255,255,0]
	lane_color_2 =tf.constant([255,0,0],dtype=tf.uint8)
	lane_color_3 =tf.constant([0,0,255],dtype=tf.uint8)
	lane_color_4 =tf.constant([0,255,0],dtype=tf.uint8)
	bg_color =tf.constant([0,0,0],dtype=tf.uint8)

	lane_line_color = tf.constant([255,255,255],dtype=tf.uint8)
	
	inp_file_name = tf.strings.join([img_dir,filename])
	image_string = tf.io.read_file(inp_file_name)
	image_decoded = tf.image.decode_png(image_string,channels=0,dtype=tf.uint8)
	image_resized = tf.image.resize_images(image_decoded, [img_h, img_w])
	
	lab_file_name = tf.strings.join([img_dir,label_bin])
	image_string = tf.io.read_file(lab_file_name)
	label_decoded = tf.image.decode_png(image_string,channels=0,dtype=tf.uint8)
	label_resized = tf.cast(tf.image.resize_images(label_decoded, [img_h, img_w]),dtype=tf.uint8)

	lab_ins_file_name = tf.strings.join([img_dir,label_ins])
	image_ins_string = tf.io.read_file(lab_ins_file_name)
	label_ins_decoded = tf.image.decode_png(image_ins_string,channels=0,dtype=tf.uint8)

	label_ins_resized = tf.cast(tf.image.resize_images(label_ins_decoded, [img_h, img_w]),dtype=tf.uint8)

	# road_im1 = tf.cast(tf.reduce_all(tf.equal(tf.cast(tf.image.resize_images(label_decoded, [img_h, img_w]),dtype=tf.uint8),lane_color_1),axis=2),dtype=tf.uint8)
	# road_im2 = tf.cast(tf.reduce_all(tf.equal(tf.cast(tf.image.resize_images(label_decoded, [img_h, img_w]),dtype=tf.uint8),lane_color_2),axis=2),dtype=tf.uint8)
	# road_im3 = tf.cast(tf.reduce_all(tf.equal(tf.cast(tf.image.resize_images(label_decoded, [img_h, img_w]),dtype=tf.uint8),lane_color_3),axis=2),dtype=tf.uint8)
	# road_im4 = tf.cast(tf.reduce_all(tf.equal(tf.cast(tf.image.resize_images(label_decoded, [img_h, img_w]),dtype=tf.uint8),lane_color_4),axis=2),dtype=tf.uint8)
	# road_im1 = tf.cast(tf.reduce_all(tf.equal(label_resized,lane_color_1),axis=2),dtype=tf.uint8)
	# road_im2 = tf.cast(tf.reduce_all(tf.equal(label_resized,lane_color_2),axis=2),dtype=tf.uint8)
	# road_im3 = tf.cast(tf.reduce_all(tf.equal(label_resized,lane_color_3),axis=2),dtype=tf.uint8)
	# road_im4 = tf.cast(tf.reduce_all(tf.equal(label_resized,lane_color_4),axis=2),dtype=tf.uint8)


	road_line_im1 = tf.cast(tf.reduce_all(tf.equal(label_resized,lane_line_color),axis=2),dtype=tf.uint8)
	'''
	bg_im = tf.reduce_all(tf.not_equal(label_resized,lane_color_1),axis=2)
	bg_im1 = tf.reduce_all(tf.not_equal(label_resized,lane_color_2),axis=2)
	temp = tf.math.logical_and(bg_im,bg_im1)
	bg_im2 =tf.reduce_all(tf.not_equal(label_resized,lane_color_3),axis=2)
	bg_im3 =tf.reduce_all(tf.not_equal(label_resized,lane_color_4),axis=2)
	temp_1 = tf.math.logical_and(bg_im2,bg_im3)
	temp_final = tf.cast(tf.math.logical_and(temp,temp_1),dtype=tf.uint8)

	check_var=tf.cast(bg_im,dtype=tf.uint8) 
	'''

	bg_img = tf.cast(tf.reduce_all(tf.equal(label_resized,bg_color),axis=2),dtype=tf.uint8)
	gt_image=tf.cast(tf.stack((road_line_im1,bg_img),axis=2),dtype=tf.float32) #,road_im2,road_im3,road_im4

 	
 	



	# print (label)
	# label = tf.reshape(label,[5,1])
	return image_resized, gt_image,label_ins_resized



def image_data_read(file_names_inp,labels_inp,labels_ins_inp,img_dir_path,img_h,img_w):
	# A vector of filenames.

	file_names = np.array(file_names_inp).reshape(file_names_inp.shape[0],)
	# print "))))))))))))))))))))))))))))):::::",file_names.shape
	labels_bin = np.array(labels_inp).reshape(labels_inp.shape[0],)
	labels_ins = np.array(labels_inp).reshape(labels_inp.shape[0],)
	# filenames = tf.constant(file_names)
	# global img_dir
	img_dir = np.array(img_dir_path).reshape(1,)
	img_dir = tf.constant( img_dir , shape=[file_names_inp.shape[0],] )
	# coords_tf = tf.constant( coords_vals , shape=[coords_vals.shape[0],coords_vals.shape[1],coords_vals.shape[2]] )
	# `labels[i]` is the label for the image in `filenames[i].
	# print labels
	# labels = tf.constant(labels)

	dataset = tf.data.Dataset.from_tensor_slices((file_names, labels_bin,labels_ins,img_dir))
	dataset = dataset.map(_parse_function)
	return dataset



def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
	"""
	Generate training image give image paths and associated steering angles
	"""

	images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
	steers = np.empty(batch_size)
	while True:
		i = 0
		# print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
		for index in np.random.permutation(image_paths.shape[0]):
			# print ("TOTAL IMG:",image_paths.shape,"\n Total Steer:",len(steering_angles))
			# print (index)
			center, left, right = image_paths[index]
			# print (center)
			steering_angle = steering_angles[index]
			# print (steering_angle)
			# argumentation
			# print ("IS Training",is_training)
			if is_training and np.random.rand() < 0.6:
				image, steering_angle = augument(data_dir, center, left, right, steering_angle)
				# print (steering_angle)
			else:
				image = load_image(data_dir, center) 
			# print ("before preprocessing")
			# add the image and steering angle to the batch
			images[i] = preprocess(image)
			# print("AFTER Preprocessing")
			steers[i] = steering_angle
			i += 1

			if i == batch_size:
				break
		# print("___________________________________________________________________________________________________",index)
		yield images, steers

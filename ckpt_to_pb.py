import tensorflow as tf 
from tensorflow.python.compiler.tensorrt import trt_convert as trt

parent_dir  = "RUNS/"
model_name="model_495"
meta_path=parent_dir+model_name+'.ckpt.meta'
output_name = parent_dir+model_name+'_v1.0.pb'
output_node_names=['Input/X','Binary_Seg/FULL_CONV_binary','Instance_seg/FULL_CONV_instance']

with tf.Session() as sess:
	saver = tf.train.import_meta_graph(meta_path)
	RUNS_FILE=parent_dir+model_name+".ckpt"
	saver.restore(sess,RUNS_FILE)
	graph_def=tf.get_default_graph().as_graph_def()
	node_list=[n.name for n in graph_def.node]


	for node in graph_def.node:
		if node.op == 'RefSwitch':
			node.op = 'Switch'
			for index in range(len(node.input)):
				if 'moving_' in node.input[index]:
					node.input[index] = node.input[index] + '/read'
		elif node.op == 'AssignSub':
			node.op = 'Sub'
			if 'use_locking' in node.attr: del node.attr['use_locking']
	# for n in graph_def.node:
	# 	if "Input" in n
	# print ([n.name for n in graph_def.node if "Input" in n.name ] )
	# for node in graph_def.node:
		
	# 	if node.op == 'RefSwitch':
	# 		# print ("------------------------------------------------")
	# 		node.op = 'Switch'
	# 		for index in range(len(node.input)):
	# 			# print node.input[index]
	# 			if 'Moving' in node.input[index] and "Switch" in node.input[index]:
	# 				# print node
	# 				print (node.input[index])
	# 				node.input[index] = node.input[index] + '/read'
	# 	elif node.op == 'AssignSub':
	# 		# print ("sub identified")
	# 		node.op = 'Sub'
	# 		if 'use_locking' in node.attr: del node.attr['use_locking']
	# 	elif node.op == 'AssignAdd':
	# 		# print ("add identified")
	# 		node.op = 'Add'
	# 		if 'use_locking' in node.attr: del node.attr['use_locking']
	# for i in node_list:
		# print i
		# if "Decoder" in i:
		# 	print i
		# if "Input" in i:
		# 	print i
		# if "FULL_CONV" in i:
		# 	print i

	frozen_graph_def=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names)
	print (output_name)
	with open(output_name,'wb')as f:
		f.write(frozen_graph_def.SerializeToString())
	# trt_graph = trt.create_inference_graph(
	# 	input_graph_def=frozen_graph_def,  # Pass the parsed graph def here
	# 	outputs=['Binary_Seg/FULL_CONV_binary','Instance_seg/FULL_CONV_instance'],
	# 	max_batch_size=1,
	# 	# max_workspace_size_bytes=1 << 25,
	# 	precision_mode='FP16',
	# 	minimum_segment_size=50
	# )

	# tf.io.write_graph(trt_graph, "frozen_models/",
	# 					"trt_model_101.pb", as_text=False)

	# tf.io.write_graph(trt_graph, "frozen_models/",
	# 					"trt_model_101.txt", as_text=True)

# %%

# 	converter = trt.TrtGraphConverter(input_graph_def=frozen_graph_def,nodes_blacklist=['Binary_Seg/FULL_CONV_binary','Instance_seg/FULL_CONV_instance'])
# 	trt_graph = converter.convert()
# 	# output_node = tf.import_graph_def(trt_graph,return_elements=['Binary_Seg/FULL_CONV_binary','Instance_seg/FULL_CONV_instance'])
# 	# sess.run(output_node)
# 	output_name = "frozen_models/"+model_name+'_v1.0.pb'

# 	with open(output_name,'wb')as f:
# 		f.write(trt_graph.SerializeToString())

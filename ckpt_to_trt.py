
# %%
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

parent_dir = "RUNS/"
model_name = "model_101"
meta_path = parent_dir+model_name+'.ckpt.meta'
output_name = "frozen_models/"+model_name+'.pb'
output_node_names = [ 'Binary_Seg/logits_to_softmax']
# ['Input/X','Binary_Seg/FULL_CONV_binary']
# ,'Instance_seg/FULL_CONV_instance']

with tf.Session() as sess:
	saver = tf.train.import_meta_graph(meta_path)
	RUNS_FILE = parent_dir+model_name+".ckpt"
	saver.restore(sess, RUNS_FILE)
	graph_def = tf.get_default_graph().as_graph_def()

	for node in graph_def.node:
		if node.op == 'RefSwitch':
			node.op = 'Switch'
			for index in range(len(node.input)):
				if 'moving_' in node.input[index]:
					node.input[index] = node.input[index] + '/read'
		elif node.op == 'AssignSub':
			node.op = 'Sub'
			if 'use_locking' in node.attr: del node.attr['use_locking']
	
	frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, output_node_names)
	print(output_name)
	converter = trt.TrtGraphConverter(input_graph_def=frozen_graph_def,nodes_blacklist=['Binary_Seg/logits_to_softmax'])
	trt_graph = converter.convert()
	output_node = tf.import_graph_def(trt_graph,return_elements=['Binary_Seg/logits_to_softmax'])
	sess.run(output_node)
	with open(output_name,'wb')as f:
		f.write(frozen_graph_def.SerializeToString())

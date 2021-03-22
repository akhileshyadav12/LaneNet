#%%
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
# import tensorflow.contrib.tensorrt as trt
# %%
frozen_graph  ='RUNS/model_495_v1_0_new.pb'

# output_names = ['conv2d_59','conv2d_67','conv2d_75']

# Read graph def (binary format)
with open(frozen_graph, 'rb') as f:
    frozen_graph_gd = tf.GraphDef()
    frozen_graph_gd.ParseFromString(f.read())


trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph_gd,  # Pass the parsed graph def here
    outputs=['Input/X','Binary_Seg/FULL_CONV_binary','Instance_seg/FULL_CONV_instance'],
    max_batch_size=1,
    # max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

tf.io.write_graph(trt_graph, "frozen_models/",
                     "trt_model_101.pb", as_text=False)

tf.io.write_graph(trt_graph, "frozen_models/",
                     "trt_model_101.txt", as_text=True)


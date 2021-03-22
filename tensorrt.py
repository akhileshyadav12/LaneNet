#%%
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
# model_filename ='RUNS_data_town_10_hd_aug/model_51_v1.0.pb'

# frozen_graph = r"\road_region_custom_run\RUNS\model_51_v1.0.pb"
# #%%
# converter = trt.TrtGraphConverter(input_graph_def=model_filename,)
#                                 #   es_blacklist=['logits', 'classes'])
# frozen_graph = converter.convert()
# output_saved_model_dir = r"\road_region_custom_run\RUNS\model_51_trt.pb"
# converter.save(output_saved_model_dir)

#%%
# import tensorflow.contrib.tensorrt as trt
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# %%
frozen_graph  ='frozen_models/model_101.pb'

# output_names = ['conv2d_59','conv2d_67','conv2d_75']

# Read graph def (binary format)
with open(frozen_graph, 'rb') as f:
    frozen_graph_gd = tf.GraphDef()
    frozen_graph_gd.ParseFromString(f.read())
#%%    

# If frozen graph is in text format load it like this
# import google.protobuf.text_format
# with open(frozen_graph, 'r') as f:
#     frozen_graph_gd = google.protobuf.text_format.Parse(f.read(), tf.GraphDef())

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph_gd,  # Pass the parsed graph def here
    outputs=['Binary_Seg/logits_to_softmax'],
    max_batch_size=1,
    # max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

tf.io.write_graph(trt_graph, "frozen_models/",
                     "trt_model_101.pb", as_text=False)

tf.io.write_graph(trt_graph, "frozen_models/",
                     "trt_model_101.txt", as_text=True)

# %%
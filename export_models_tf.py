import onnx
from onnx_tf.backend import prepare
print("The ONNX Model Saved")
onnx_model = onnx.load('lipreading.onnx')
print("The ONNX Model Loaded")
tf_rep = prepare(onnx_model) 
print("The ONNX Model Prepared for TF")
tf_rep.export_graph('1')
print("The ONNX Model Exported to TF")

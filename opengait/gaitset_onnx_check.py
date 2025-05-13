import onnx
import onnxruntime as ort
import numpy as np

onnx_model = onnx.load("gaitset_model.onnx")
onnx.checker.check_model(onnx_model)

ort_session = ort.InferenceSession("gaitset_model.onnx")

dummy_input = np.random.randn(1, 1, 30, 64, 44).astype(np.float32)
outputs = ort_session.run(None, {'input': dummy_input})
print(outputs[0].shape)

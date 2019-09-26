import syft
from syft_tensorflow.hook import TensorFlowHook

setattr(syft, "TensorFlowHook", TensorFlowHook)

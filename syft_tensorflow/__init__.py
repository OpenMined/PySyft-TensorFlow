from syft_tensorflow.serde import MAP_TF_SIMPLIFIERS_AND_DETAILERS
from syft_tensorflow.tensor import TensorFlowTensor


def bind_tensorflow(*module):
    from syft_tensorflow.hook import TensorFlowHook

    for m in module:
        setattr(m, "TensorFlowHook", TensorFlowHook)

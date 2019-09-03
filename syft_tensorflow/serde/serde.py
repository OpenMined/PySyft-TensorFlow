from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

import syft

from syft.generic.tensor import initialize_tensor


def _simplify_tf_tensor(tensor: tf.Tensor) -> bin:
    """
    This function converts a TF tensor into a serialized TF tensor
    using tf.io. We do this because it's native to TF, and they've optimized it.

    Args:
      tensor (torch.Tensor): an input tensor to be serialized

    Returns:
      tuple: serialized tuple of torch tensor. The first value is the
      id of the tensor and the second is the binary for the PyTorch
      object. The third is the chain of abstractions, and the fourth
      (optinally) is the chain of graident tensors (nested tuple)
    """

    tensor_ser = tf.io.serialize_tensor(tensor)

    dtype_ser = tensor.dtype.as_datatype_enum

    chain = None
    if hasattr(tensor, "child"):
        chain = syft.serde._simplify(tensor.child)

    return tensor.id, tensor_ser.numpy(), dtype_ser, chain


def _detail_tf_tensor(worker, tensor_tuple) -> tf.Tensor:
    """
    This function converts a serialized tf tensor into a local tf tensor
    using tf.io.

    Args:
        tensor_tuple (bin): serialized obj of torch tensor. It's a tuple where
            the first value is the ID, the second vlaue is the binary for the
            TensorFlow object, the third value is the tensor_dtype_enum, and
            the fourth value is the chain of tensor abstractions

    Returns:
        tf.Tensor: a deserialized TF tensor
    """

    tensor_id, tensor_bin, tensor_dtype_enum, chain = tensor_tuple

    tensor_dtype = tf.dtypes.DType(tensor_dtype_enum)
    tensor = tf.io.parse_tensor(tensor_bin, tensor_dtype)

    initialize_tensor(
        hook_self=syft.tensorflow.hook,
        cls=tensor,
        is_tensor=True,
        owner=worker,
        id=tensor_id,
        init_args=[],
        kwargs={},
    )

    if chain is not None:
        chain = syft.serde._detail(worker, chain)
        tensor.child = chain
        tensor.is_wrapper = True

    return tensor


MAP_TF_SIMPLIFIERS_AND_DETAILERS = OrderedDict(
    {
        EagerTensor: (_simplify_tf_tensor, _detail_tf_tensor),
        tf.Tensor: (_simplify_tf_tensor, _detail_tf_tensor),
    }
)

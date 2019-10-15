from collections import OrderedDict
import h5py
import io
from tempfile import TemporaryDirectory

import syft
from syft.generic.object import initialize_object
from syft.generic.tensor import initialize_tensor
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable


def _simplify_tf_tensor(tensor: tf.Tensor) -> bin:
    """
    This function converts a TF tensor into a serialized TF tensor using
    tf.io. We do this because it's native to TF, and they've optimized it.

    Args:
      tensor (tf.Tensor): an input tensor to be serialized

    Returns:
      tuple: serialized tuple of TensorFlow tensor. The first value is the
      id of the tensor and the second is the binary for the TensorFlow
      object. The third is the tensor dtype and the fourth is the chain
      of abstractions.
    """

    tensor_ser = tf.io.serialize_tensor(tensor)

    dtype_ser = tensor.dtype.as_datatype_enum

    chain = None
    if hasattr(tensor, "child"):
        chain = syft.serde._simplify(tensor.child)

    return tensor.id, tensor_ser.numpy(), dtype_ser, chain


def _detail_tf_tensor(worker, tensor_tuple) -> tf.Tensor:
    """
    This function converts a serialized tf tensor into a local TF tensor
    using tf.io.

    Args:
        tensor_tuple (bin): serialized obj of TF tensor. It's a tuple where
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
        hook=syft.tensorflow.hook,
        obj=tensor,
        owner=worker,
        id=tensor_id,
        init_args=[],
        init_kwargs={},
    )

    if chain is not None:
        chain = syft.serde._detail(worker, chain)
        tensor.child = chain
        tensor.is_wrapper = True

    return tensor


def _simplify_tf_variable(tensor: tf.Variable) -> bin:
    """
    This function converts a TF variable into a serialized TF variable using
    tf.io. We do this because it's native to TF, and they've optimized it.

    Args:
      tensor (tf.Variable): an input variable to be serialized

    Returns:
      tuple: serialized tuple of TensorFlow tensor. The first value is the
      id of the tensor and the second is the binary for the TensorFlow
      object. The third is the tensor dtype and the fourth is the chain
      of abstractions.
    """

    tensor_ser = tf.io.serialize_tensor(tensor)

    dtype_ser = tensor.dtype.as_datatype_enum

    chain = None
    if hasattr(tensor, "child"):
        chain = syft.serde._simplify(tensor.child)

    return tensor.id, tensor_ser.numpy(), dtype_ser, chain


def _detail_tf_variable(worker, tensor_tuple) -> tf.Tensor:
    """
    This function converts a serialized TF variable into a local TF tensor
    using tf.io.

    Args:
        tensor_tuple (bin): serialized obj of TF variable. It's a tuple where
            the first value is the ID, the second vlaue is the binary for the
            TensorFlow object, the third value is the tensor_dtype_enum, and
            the fourth value is the chain of tensor abstractions

    Returns:
        tf.Tensor: a deserialized TF tensor
    """

    tensor_id, tensor_bin, tensor_dtype_enum, chain = tensor_tuple

    tensor_dtype = tf.dtypes.DType(tensor_dtype_enum)
    tensor = tf.io.parse_tensor(tensor_bin, tensor_dtype)
    tensor = tf.Variable(tensor)

    initialize_tensor(
        hook=syft.tensorflow.hook,
        obj=tensor,
        owner=worker,
        id=tensor_id,
        init_args=[],
        init_kwargs={},
    )

    if chain is not None:
        chain = syft.serde._detail(worker, chain)
        tensor.child = chain
        tensor.is_wrapper = True

    return tensor


def _simplify_tf_keras_layers(layer: tf.Tensor) -> bin:
    """
    This function converts a keras layer into a serialized keras layer using
    tf.io. We do this because it's native to TF, and they've optimized it.

    Args:
      tensor (torch.Tensor): an input tensor to be serialized

    Returns:
      tuple: serialized tuple of TensorFlow tensor. The first value is the
      id of the tensor and the second is the binary for the TensorFlow
      object. The third is the tensor dtype and the fourth is the chain
      of abstractions.
    """

    layer_ser = tf.keras.layers.serialize(layer)

    weights = layer.get_weights()
    weights_ser = syft.serde.serde._simplify(weights)

    layer_dict_ser = syft.serde.serde._simplify(layer_ser)

    batch_input_shape_ser = syft.serde.serde._simplify(
        layer_ser["config"]["batch_input_shape"]
    )

    chain = None
    if hasattr(layer, "child"):
        chain = syft.serde._simplify(layer.child)

    return layer.id, layer_dict_ser, weights_ser, batch_input_shape_ser, chain


def _detail_tf_keras_layers(worker, layer_tuple) -> tf.Tensor:
    """
    This function converts a serialized tf tensor into a local TF tensor
    using tf.io.

    Args:
        tensor_tuple (bin): serialized obj of torch tensor. It's a tuple where
            the first value is the ID, the second vlaue is the binary for the
            TensorFlow object, the third value is the tensor_dtype_enum, and
            the fourth value is the chain of tensor abstractions

    Returns:
        tf.Tensor: a deserialized TF tensor
    """

    layer_id, layer_bin, weights_bin, batch_input_shape_bin, chain = layer_tuple

    layer_dict = syft.serde.serde._detail(worker, layer_bin)

    layer = tf.keras.layers.deserialize(layer_dict)

    weights = syft.serde.serde._detail(worker, weights_bin)

    batch_input_shape = syft.serde.serde._detail(worker, batch_input_shape_bin)

    layer.build(batch_input_shape)

    layer.set_weights(weights)

    initialize_object(
        hook=syft.tensorflow.hook,
        obj=layer,
        owner=worker,
        reinitialize=False,
        id=layer_id,
        init_args=[],
        init_kwargs={},
    )

    if chain is not None:
        chain = syft.serde._detail(worker, chain)
        layer.child = chain
        layer.is_wrapper = True

    return layer


def _simplify_keras_model(model: tf.keras.Model):
    bio = io.BytesIO()

    with h5py.File(bio) as file:
        tf.keras.models.save_model(model, file, include_optimizer=True, overwrite=True)

    model_ser = bio.getvalue()

    return model_ser, model.id


def _detail_keras_model(worker, model_tuple):
    model_ser, model_id = model_tuple
    bio = io.BytesIO(model_ser)
    with h5py.File(bio) as file:
        model = tf.keras.models.load_model(file)

    initialize_object(
        hook=syft.tensorflow.hook,
        obj=model,
        owner=worker,
        reinitialize=False,
        id=model_id,
        init_args=[],
        init_kwargs={},
    )

    return model


def _simplify_tf_tensorshape(tensorshape: tf.TensorShape) -> bin:
    """
    This function converts a TF tensor shape into a serialized list.

    Args:
      tensor (tf.TensorShape): an input tensor shape to be serialized

    Returns:
      tuple: serialized tuple of TF tensor shape. The first value is
      the binary for the TensorShape object. The second is the
      chain of abstractions.
    """

    tensorshape_list_ser = syft.serde.serde._simplify(tensorshape.as_list())

    # TODO[Yann] currently this condition is never true,
    # tf.TensorShape needs to be hooked
    chain = None
    if hasattr(tensorshape, "child"):
        chain = syft.serde._simplify(tensorshape.child)

    return tensorshape_list_ser, chain


def _detail_tf_tensorshape(worker, tensor_tuple) -> tf.TensorShape:
    """
    This function converts a serialized TF tensor shape into a local list.

    Args:
        tensor_tuple (bin): serialized obj of TF tensor shape as list.
            It's a tuple where the first value is the binary for the
            tensorflow shape (as list), and the third value is the
            chain of tensor abstractions.

    Returns:
        tf.Tensor: a deserialized TF tensor
    """

    tensorshape_list_bin, chain = tensor_tuple

    tensorshape_list = syft.serde.serde._detail(worker, tensorshape_list_bin)

    tensorshape = tf.TensorShape(tensorshape_list)

    # TODO[Yann] currently this condition is never true,
    # tf.TensorShape needs to be hooked
    if chain is not None:
        chain = syft.serde._detail(worker, chain)
        tensorshape.child = chain
        tensorshape.is_wrapper = True

    return tensorshape


MAP_TF_SIMPLIFIERS_AND_DETAILERS = OrderedDict(
    {
        EagerTensor: (_simplify_tf_tensor, _detail_tf_tensor),
        tf.Tensor: (_simplify_tf_tensor, _detail_tf_tensor),
        tf.TensorShape: (_simplify_tf_tensorshape, _detail_tf_tensorshape),
        tf.Variable: (_simplify_tf_variable, _detail_tf_variable),
<<<<<<< HEAD
=======
        tf.keras.layers.Layer: (_simplify_tf_keras_layers, _detail_tf_keras_layers),
        tf.keras.models.Model: (_simplify_keras_model, _detail_keras_model),
>>>>>>> big spaghetti
        ResourceVariable: (_simplify_tf_variable, _detail_tf_variable),
        tf.keras.layers.Layer: (_simplify_tf_keras_layers, _detail_tf_keras_layers)
    }
)

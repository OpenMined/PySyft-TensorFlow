import pytest
import tensorflow as tf
import syft
import numpy as np


def test_send_get_keras_layer(remote):

    layer_to_give =  tf.keras.layers.Dense(5, input_shape=[2])
    layer_to_give.build([2])

    layer_ptr = layer_to_give.send(remote)
    layer_gotten = layer_ptr.get()

    np.array_equal(layer_to_give.get_weights()[0],
                   layer_gotten.get_weights()[0])

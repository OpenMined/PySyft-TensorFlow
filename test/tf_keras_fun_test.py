import pytest
import tensorflow as tf

import syft


def test_keras_activations_fn():
    hook = syft.TensorFlowHook(tf)
    bob = syft.VirtualWorker(hook, id="bob")

    x_to_give = tf.constant([-2.0, 3.0, 5.0])
    x_ptr = x_to_give.send(bob)
    
    relu_ptr = tf.keras.activations.relu(x_ptr)
import pytest
import tensorflow as tf

import syft


def test_send_get_constant():
    hook = syft.TensorFlowHook(tf)
    bob = syft.VirtualWorker(hook, id="bob")

    x_to_give = tf.constant(2.0)
    x_ptr = x_to_give.send(bob)
    x_gotten = x_ptr.get()
    assert tf.math.equal(x_to_give, x_gotten)

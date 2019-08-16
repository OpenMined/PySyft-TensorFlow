import tensorflow as tf

import syft


def test_serde_constant():
    hook = syft.TensorFlowHook(tf)
    syft.tensorflow.hook = hook

    z = tf.constant([1.0, 2.0])
    z.id = 123456

    ser = syft.serde.serialize(z)
    x = syft.serde.deserialize(ser)

    assert all(tf.math.equal(x, z))
    assert x.id == z.id
    assert x.dtype == z.dtype

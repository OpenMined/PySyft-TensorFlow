import tensorflow as tf
import numpy as np


def test_send_get_variable(remote):
  x_to_give = tf.Variable(2.0)
  x_ptr = x_to_give.send(remote)
  x_gotten = x_ptr.get()
  assert np.array_equal(x_to_give.numpy(), x_gotten.numpy())
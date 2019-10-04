import tensorflow as tf

def test_variable_add(remote):
  #breakpoint()
  x = tf.Variable(5)
  x = tf.Variable(5).send(remote)
  y = tf.Variable(5).send(remote)

  z = x + y

  print(z.get())
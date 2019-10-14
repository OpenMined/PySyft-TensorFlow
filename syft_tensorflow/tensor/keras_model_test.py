import pytest
import tensorflow as tf
import syft
import numpy as np


def test_send_get_keras_model(remote):

    model_to_give = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(5, input_shape=[2])
    ])

    model_ptr = model_to_give.send(remote)
    model_gotten = model_ptr.get()

    assert np.array_equal(model_to_give.get_weights()[0],
                model_gotten.get_weights()[0])


def test_keras_sequential(remote):

    x_to_give = tf.random.uniform([2, 2])
    model_to_give = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(5, input_shape=[2])
    ])
    expected = model_to_give(x_to_give)

    x_ptr = x_to_give.send(remote)
    model_ptr = model_to_give.send(remote)
    actual = model_ptr(x_ptr).get()

    assert np.array_equal(actual, expected)


def test_keras_model_compile(remote):

    model_to_give = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(5, input_shape=[2])
    ])

    model_ptr = model_to_give.send(remote)

    model_ptr.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model_on_worker = remote._objects[model_ptr.id_at_location]

    assert model_on_worker.loss == 'categorical_crossentropy'
    assert isinstance(model_on_worker.optimizer,
            tf.keras.optimizers.Adam)





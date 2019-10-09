import inspect
import logging
import types

import tensorflow as tf
from tensorflow.python.framework.ops import Tensor

import syft
from syft.workers.base import BaseWorker
from syft.workers.virtual import VirtualWorker
from syft.generic.frameworks.hook.hook import FrameworkHook
from syft.generic.tensor import initialize_tensor

from syft_tensorflow.attributes import TensorFlowAttributes
from syft_tensorflow.tensor import TensorFlowTensor
from syft_tensorflow.tensor import TensorFlowVariable
from syft_tensorflow.tensor import KerasLayer


class TensorFlowHook(FrameworkHook):
    def __init__(
        self,
        tensorflow,
        local_worker: BaseWorker = None,
        is_client: bool = True
    ):

        self.tensorflow = tensorflow
        self.framework = self.tensorflow

        syft.tensorflow = TensorFlowAttributes(tf, self)

        syft.framework = syft.tensorflow
        syft.tensorflow.hook = self
        syft.hook = self

        self.local_worker = local_worker

        if hasattr(tensorflow, "tf_hooked"):
            logging.warning("TF was already hooked, skipping hooking process")
            self.local_worker = syft.local_worker
            return
        else:
            tensorflow.tf_hooked = True

        if self.local_worker is None:
            # Every TensorFlowHook instance should have a local worker which is
            # responsible for interfacing with other workers. The worker
            # interface is what allows the TensorFlow specific code in TensorFlowHook to
            # be agnostic to the means by which workers communicate (such as
            # peer-to-peer, sockets, through local ports, or all within the
            # same process)
            self.local_worker = VirtualWorker(
                hook=self, is_client_worker=is_client, id="me"
            )
        else:
            self.local_worker.hook = self

        self.to_auto_overload = {
          Tensor: self._which_methods_should_we_auto_overload(
              Tensor
          ),

          tf.Variable: self._which_methods_should_we_auto_overload(
              tf.Variable
          ),

          tf.keras.layers.Layer: self._which_methods_should_we_auto_overload(
              tf.keras.layers.Layer
          ),

        }

        self.args_hook_for_overloaded_attr = {}

        self._hook_native_tensor(Tensor, TensorFlowTensor)
        self._hook_native_tensor(tf.Variable, TensorFlowVariable)

        self._hook_native_keras_objects(tf.keras.layers.Layer, KerasLayer)

        self._hook_pointer_tensor_methods(Tensor)
        self._hook_pointer_tensor_methods(tf.Variable)
        self._hook_pointer_tensor_methods(tf.keras.layers.Layer)

        self._hook_tensorflow_module()

        syft.local_worker = self.local_worker
        syft.hook = self

        # This must happen last!
        # See this functions documentation for more info.
        self._add_methods_to_eager_tensor()

    def _hook_native_tensor(self, tensor_type: type, syft_type: type):
        """Adds PySyft Tensor Functionality to the given native tensor type.
         Overloads the given native TensorFlow tensor to add PySyft Tensor
        Functionality. Overloading involves modifying the tensor type with
        PySyft's added functionality. You may read about what kind of
        modifications are made in the methods that this method calls.
         Args:
            tensor_type: The type of tensor being hooked (in this refactor
                this is only ever tf.Tensor, but in previous versions of
                PySyft this iterated over all tensor types.
            syft_type: The abstract type whose methods should all be added to
                the tensor_type class. In practice this is always TensorFlowTensor.
                Read more about it there.
        """
        # Reinitialize init method of TensorFlow tensor with Syft init
        self._add_registration_to___init__(tensor_type)

        # Overload TensorFlow tensor properties with Syft properties
        self._hook_properties(tensor_type)

        # Overload auto overloaded with TensorFlow methods
        self._add_methods_from_native_tensor(tensor_type, syft_type)

        self._hook_native_methods(tensor_type)

    def _hook_native_keras_objects(self, tensor_type: type, syft_type: type):

        # Reinitialize init method of Torch tensor with Syft init
        self._add_registration_to___init__(tensor_type)

        # Overload Torch tensor properties with Syft properties
        self._hook_properties(tensor_type)

        # Overload auto overloaded with Torch methods
        self._add_methods_from_native_tensor(tensor_type, syft_type)

        self._hook_layer_methods(tensor_type)


    def _hook_tensorflow_module(self):
        tensorflow_modules = syft.tensorflow.tensorflow_modules

        for module_name, tensorflow_module in tensorflow_modules.items():
            for func in dir(tensorflow_module):

                # Some functions we want to ignore (not override). Such functions have been hard
                # coded into the tensorflow_attribute exclude (see TensorFlowAttribute class)
                if func in syft.tensorflow.exclude:
                    continue

                # ignore dunder functions
                if "__" in func:
                    continue

                # ignore capitalized func values which are Classes not functinos
                if func[0].isupper():
                    continue

                # ignore hidden functins
                if func[0] == "_":
                    continue

                # If we haven't already overloaded this function
                if "native_" in func or f"native_{func}" in dir(tensorflow_module):
                    continue

                self._perform_function_overloading(module_name, tensorflow_module, func)

    def _add_registration_to___init__(
        hook_self, tensor_type: type, is_tensor: bool = False
    ):
        """Adds several attributes to the tensor.
         Overloads tensor_type.__init__ to add several attributes to the tensor
        as well as (optionally) registering the tensor automatically.
        TODO: auto-registration is disabled at the moment, this might be bad.
         Args:
            tensor_type: The type of tensor being hooked (in this refactor this
                is only ever tf.Tensor, but in previous versions of PySyft
                this iterated over all tensor types.
            is_tensor: An optional boolean parameter (default False) to
                specify whether to skip running the native initialization
                logic. TODO: this flag might never get used.
        """

        def new___init__(cls, *args, owner=None, id=None, register=True, **kwargs):
            cls.native___init__(*args, **kwargs)

            if owner is None:
              owner = hook_self.local_worker

            if id is None:
              id = syft.ID_PROVIDER.pop()

            cls.id = id
            cls.owner = owner
            cls.is_wrapper = False

        if "native___init__" not in dir(tensor_type):
            tensor_type.native___init__ = tensor_type.__init__

        tensor_type.__init__ = new___init__

    @staticmethod
    def _add_methods_from_native_tensor(tensor_type: type, syft_type: type):
        """Adds methods from the TensorFlowTensor class to the native TensorFlow tensor.
         The class TensorFlowTensor is a proxy to avoid extending directly the TensorFlow
        tensor class.
         Args:
            tensor_type: The tensor type to which we are adding methods
                from TensorFlowTensor class.
        """
        exclude = [
            "__class__",
            "__delattr__",
            "__dir__",
            "__doc__",
            "__dict__",
            "__format__",
            "__getattribute__",
            "__hash__",
            "__init__",
            "__init_subclass__",
            "__weakref__",
            "__ne__",
            "__new__",
            "__reduce__",
            "__reduce_ex__",
            "__setattr__",
            "__sizeof__",
            "__subclasshook__",
            # "__eq__", # FIXME see PySyft
            "__gt__",
            "__ge__",
            "__lt__",
            "__le__",
        ]
        # For all methods defined in tf.Tensor or TensorFlowTensor
        # that are not internal methods (like __class__etc)
        for attr in dir(syft_type):
            if attr not in exclude:
                # Alias `attr` method as `native_attr` if it already exists
                if hasattr(tensor_type, attr):
                    setattr(
                        tensor_type,
                        f"native_{attr}",
                        getattr(tensor_type, attr)
                    )
                # Add this method to the TF tensor
                setattr(tensor_type, attr, getattr(syft_type, attr))

    def _add_methods_to_eager_tensor(self):
      """
      Add required TensorFlowTensor methods to EagerTensor.

      When a user creates a tensor, e.g. with `tf.constant([1,2,3])`, the
      type of the returned object is an `EagerTensor`.  EagerTensor is defined
      in tensorflow and is a super class of tf.Tensor.  However, EagerTensor is
      not importable so we cannot add the properties to it that we want.
      So we do that here, which requires us to instantiate an instance of an
      EagerTensor to then get reference to the type.

      We do it this way for 2 reasons:

      1. Avoid monkeypatching functions per instance (e.g. having to overwrite the
         function every time we make a tensor)
      2. Most dunder methods are actually looked up on the class itself, rather
         than the instance, so monkeypatching the instance doesn't even work for
         things like __repr__.
      """

      dummy = tf.constant(0)
      eager_type = type(dummy)

      eager_type.__repr__ = TensorFlowTensor.__repr__
      eager_type.__str__ = TensorFlowTensor.__str__

    def _hook_layer_methods(self, tensor_type: type):
        """
        Add hooked version of all methods of to_auto_overload[tensor_type]
        to the tensor_type; instead of performing the native tensor
        method, the hooked version will be called

        Args:
            tensor_type: the tensor_type which holds the methods
        """
        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[tensor_type]:
            # if we haven't already overloaded this function
            if (attr=='call') & (f"native_{attr}" not in dir(tensor_type)):
                native_method = getattr(tensor_type, attr)
                setattr(tensor_type, f"native_{attr}", native_method)
                new_method = self._get_hooked_method(attr)
                setattr(tensor_type, attr, new_method)

    @classmethod
    def create_shape(cls, shape_dims):
        return tf.TensorShape(shape_dims)

    @classmethod
    def create_zeros(shape, dtype, **kwargs):
        return tf.zeros(shape, dtype=dtype, **kwargs)

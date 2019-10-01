import logging
import types

import tensorflow as tf
from tensorflow.python.framework.ops import Tensor

import syft
from syft.workers.base import BaseWorker
from syft.workers.virtual import VirtualWorker
from syft.generic.frameworks.hook.hook import FrameworkHook
from syft.generic.tensor import initialize_tensor

from syft_tensorflow.tensor import TensorFlowTensor


class TensorFlowHook(FrameworkHook):
    def __init__(
        self, tensorflow, local_worker: BaseWorker = None, is_client: bool = True
    ):
        self.tensorflow = tensorflow
        self.tensorflow.hook = self
        self.framework = self.tensorflow
        syft.tensorflow = TensorFlowAttributes(tensorflow, self)
        syft.framework = syft.tensorflow
        syft.tensorflow.hook = self

        self.local_worker = local_worker

        if hasattr(tensorflow, "tf_hooked"):
            logging.warning("TF was already hooked... skipping hooking process")
            self.local_worker = syft.local_worker
            return
        else:
            tensorflow.tf_hooked = True

        if self.local_worker is None:
            # Every TorchHook instance should have a local worker which is
            # responsible for interfacing with other workers. The worker
            # interface is what allows the Torch specific code in TorchHook to
            # be agnostic to the means by which workers communicate (such as
            # peer-to-peer, sockets, through local ports, or all within the
            # same process)
            self.local_worker = VirtualWorker(
                hook=self, is_client_worker=is_client, id="me"
                
            )
        else:
            self.local_worker.hook = self

        self.to_auto_overload = {}

        self.args_hook_for_overloaded_attr = {}

        self._hook_native_tensor(Tensor, TensorFlowTensor)

        self._hook_tensorflow_module()

        syft.local_worker = self.local_worker
        syft.hook = self

    def _hook_native_tensor(self, tensor_type: type, syft_type: type):
        """Adds PySyft Tensor Functionality to the given native tensor type.
         Overloads the given native Torch tensor to add PySyft Tensor
        Functionality. Overloading involves modifying the tensor type with
        PySyft's added functionality. You may read about what kind of
        modifications are made in the methods that this method calls.
         Args:
            tensor_type: The type of tensor being hooked (in this refactor
                this is only ever torch.Tensor, but in previous versions of
                PySyft this iterated over all tensor types.
            syft_type: The abstract type whose methods should all be added to
                the tensor_type class. In practice this is always TorchTensor.
                Read more about it there.
        """
        # Reinitialize init method of Torch tensor with Syft init
        self._add_registration_to___init__(tensor_type, is_tensor=True)

        # Overload Torch tensor properties with Syft properties
        self._hook_properties(tensor_type)

        # Returns a list of methods to be overloaded, stored in the dict to_auto_overload
        # with tensor_type as a key
        # self.to_auto_overload[tensor_type] = self._which_methods_should_we_auto_overload(
        #     tensor_type
        # )

        # [We don't rename native methods as torch tensors are not hooked] Rename native functions
        # self._rename_native_functions(tensor_type)

        # Overload auto overloaded with Torch methods
        self._add_methods_from_native_tensor(tensor_type, syft_type)

        # TODO Need to add 'get_hooked_method'
        # self._hook_native_methods(tensor_type)


    def _hook_tensorflow_module(self):

        def perform_overloading(torch_module, func):

            # Where the overloading happens
            # 1. Get native function
            native_func = getattr(torch_module, func)
            # 2. Check it is a proper function
            if type(native_func) in [types.FunctionType, types.BuiltinFunctionType]:
                # 3. Build the hooked function
                new_func = self._get_hooked_func(native_func)
                # 4. Move the native function
                setattr(torch_module, f"native_{func}", native_func)
                # 5. Put instead the hooked one
                setattr(torch_module, func, new_func)

        #breakpoint()
        tensorflow_modules = syft.tensorflow.tensorflow_modules

        for module_name, tensorflow_module in tensorflow_modules.items():
            for func in dir(tensorflow_module):

                # Some functions we want to ignore (not override). Such functions have been hard
                # coded into the torch_attribute exclude (see TorchAttribute class)
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

                perform_overloading(tensorflow_module, func)

    def _add_registration_to___init__(
        hook_self, tensor_type: type, is_tensor: bool = False
    ):
        """Adds several attributes to the tensor.
         Overloads tensor_type.__init__ to add several attributes to the tensor
        as well as (optionally) registering the tensor automatically.
        TODO: auto-registration is disabled at the moment, this might be bad.
         Args:
            tensor_type: The type of tensor being hooked (in this refactor this
                is only ever torch.Tensor, but in previous versions of PySyft
                this iterated over all tensor types.
            torch_tensor: An optional boolean parameter (default False) to
                specify whether to skip running the native initialization
                logic. TODO: this flag might never get used.
        """
        if "native___init__" not in dir(tensor_type):
            tensor_type.native___init__ = tensor_type.__init__

        def new___init__(cls, *args, owner=None, id=None, register=True, **kwargs):
            initialize_tensor(
                hook_self=hook_self,
                cls=cls,
                id=id,
                is_tensor=is_tensor,
                init_args=args,
                init_kwargs=kwargs,
            )

        tensor_type.__init__ = new___init__

    def _hook_properties(hook_self, tensor_type: type):
        """Overloads tensor_type properties.

        This method gets called only on torch.Tensor. If you're not sure how
        properties work, read:
        https://www.programiz.com/python-programming/property
        Args:
            tensor_type: The tensor type which is having properties
                added to it, typically just torch.Tensor.
        """

        @property
        def location(self):
            return self.child.location

        tensor_type.location = location

        @property
        def id_at_location(self):
            return self.child.id_at_location

        tensor_type.id_at_location = id_at_location

        @property
        def id(self):
            if not hasattr(self, "_syft_id"):
                self._syft_id = syft.ID_PROVIDER.pop()
            return self._syft_id

        @id.setter
        def id(self, new_syft_id):
            self._syft_id = new_syft_id
            return self

        tensor_type.id = id

        @property
        def owner(self):
            if not hasattr(self, "_owner"):
                self._owner = hook_self.local_worker
            return self._owner

        @owner.setter
        def owner(self, new_owner):
            self._owner = new_owner
            return self

        tensor_type.owner = owner

        @property
        def is_wrapper(self):
            if not hasattr(self, "_is_wrapper"):
                self._is_wrapper = False
            return self._is_wrapper

        @is_wrapper.setter
        def is_wrapper(self, it_is_a_wrapper):
            self._is_wrapper = it_is_a_wrapper
            return self

        tensor_type.is_wrapper = is_wrapper

        tensor_type.native_shape = tensor_type.shape

        def dim(self):
            return len(self.shape)

        tensor_type.dim = dim

    @staticmethod
    def _add_methods_from_native_tensor(tensor_type: type, syft_type: type):
        """Adds methods from the TorchTensor class to the native torch tensor.
         The class TorchTensor is a proxy to avoid extending directly the torch
        tensor class.
         Args:
            tensor_type: The tensor type to which we are adding methods
                from TorchTensor class.
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
            # "__eq__", # FIXME it now overwritten in native.py to use torch.eq, because of pb between == & __eq__ See #2030
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
                    setattr(tensor_type, f"native_{attr}", getattr(tensor_type, attr))
                # Add this method to the TF tensor
                setattr(tensor_type, attr, getattr(TensorFlowTensor, attr))

    @classmethod
    def create_wrapper(cls, child_to_wrap):
        return tf.constant([])

    @classmethod
    def create_shape(cls, shape_dims):
        return tf.TensorShape(shape_dims)

    @classmethod
    def create_zeros(shape, dtype, **kwargs):
        return tf.zeros(shape, dtype=dtype, **kwargs)


from types import ModuleType

from tensorflow.python.framework.ops import EagerTensor
from syft.generic.frameworks.attributes import FrameworkAttributes

class TensorFlowAttributes(FrameworkAttributes):
    """Adds tensorflow module related custom attributes.

    TensorFlowAttributes is a special class where all custom attributes related
    to the torch module can be added. Any global parameter, configuration,
    or reference relating to PyTorch should be stored here instead of
    attaching it directly to some other part of the global namespace.

    The main reason we need this is because the hooking process occasionally
    needs to save global objects, notably including what methods to hook and
    what methods to NOT hook.

    This will hold all necessary attributes PySyft needs.

    Args:
        tensorflow: A ModuleType indicating the tensorflow module
        hook: A TensorFlowHook to stash
    """

    ALIAS = "tensorflow"
    Tensor = TensorFlowTensor

    def __init__(self, tensorflow: ModuleType, hook: TensorFlowHook):
        # Stash the hook here for global access elsewhere
        self.hook = hook

        # SECTION: List all functions in tf module that we want to overload

        # List modules that we will hook
        self.tensorflow_modules = {
            #"tensorflow": tensorflow,
            "tensorflow.keras.activations": tensorflow.keras.activations,
            # "tensorflow.keras": tensorflow.keras,
        }

        # Set of all function names with module as prefix in the modules to hook
        self.tensorflow_module_functions = {
            f"{module_name}.{func_name}"
            for module_name, tensorflow_module in self.tensorflow_modules.items()
            for func_name in dir(tensorflow_module)
        }

        # Store reference to all tf functions by string name stored in tensorflow_modules_functions
        self.eval_tensorflow_modules_functions = {
            f"{module_name}.{func_name}": getattr(tensorflow_module, func_name)
            for module_name, tensorflow_module in self.tensorflow_modules.items()
            for func_name in dir(tensorflow_module)
        }

        # Add special functions to exclude from the hook **in alphabetical order**
        # Reasons can be:
        # - Used for internal process like printing tensors
        # - Don't use tensors so are bound to have local executions
        # - etc
        # DON'T put here:
        # - functions like native_*
        # - functions that could use pointers or syft tensors
        self.exclude = []

        # SECTION: List all torch tensor methods we want to overload
        self.tensor_types = [tensorflow.Tensor, tensorflow.Variable]

        self.tensorvar_methods = list(
            {method for tensorvar in self.tensor_types for method in dir(tensorvar)}
        )
        self.tensorvar_methods += [
            "get_shape",
            "share",
            "fix_precision",
            "decode",
            "end_get",
        ]

        # SECTION: Build the guard, that define which functions or methods can be safely called by
        # external or local workers

        # Add all tensor types
        self.guard = {
            "Tensor": tensorflow.Tensor,
            "Variable": tensorflow.Variable,
            "EagerTensor": EagerTensor,
        }

        # Allow the `syft.` prefix to be used
        keys = list(self.guard.keys())
        for key in keys:
            self.guard[f"syft.{key}"] = self.guard[key]

        # Concatenate torch functions and torch methods
        self.allowed_commands = {
            "tensorvar_methods": self.tensorvar_methods,
            "framework_functions": self.tensorflow_module_functions,
        }

        # The equivalent concatenation of native torch function names and native torch method names
        self.native_commands = {
            command_type: {cmd: self.get_native_framework_name(cmd) for cmd in commands}
            for command_type, commands in self.allowed_commands.items()
        }

        self.command_guard = self._command_guard

        # Dict {method_name: <is_inplace:bool>
        self.inplace_methods = {}

    def is_inplace_method(self, method):
        # I've not yet encountered any inplace methods in TF
        return False

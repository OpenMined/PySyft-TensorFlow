import weakref

import tensorflow as tf

import syft
from syft.generic.tensor import AbstractTensor
from syft.workers.base import BaseWorker
from syft.generic.pointers.pointer_tensor import PointerTensor


class TensorFlowTensor(AbstractTensor):
    """Add methods to this tensor to have them added to every tf.Tensor object.

    This tensor is simply a more convenient way to add custom functions to
    all TensorFlow tensor types. When you add a function to this tensor, it will
    be added to EVERY native TF tensor type (i.e. tf.Tensor) automatically
    by the TensorFlowHook (which is in frameworks/tensorflow/hook.py).

    Note: all methods from AbstractTensor will also be included because this
    tensor extends AbstractTensor. So, if you're looking for a method on
    the native torch tensor API but it's not listed here, you might try
    checking AbstractTensor.
    """

    def has_child(self):
        return hasattr(self, "child")

    def describe(self, description):
        self.description = description
        return self

    def tag(self, *_tags):
        if self.tags is None:
            tags = list()
        else:
            tags = list(self.tags)

        for new_tag in _tags:
            tags.append(new_tag)

        self.tags = set(tags)
        return self

    @property
    def tags(self):
        if self.has_child():
            return self.child.tags
        else:
            if not hasattr(self, "_tags"):
                self._tags = None
            return self._tags

    @tags.setter
    def tags(self, new_tags):
        if self.has_child():
            if new_tags is not None:
                self.child.tags = set(new_tags)
            else:
                self.child.tags = set()
        else:
            self._tags = new_tags

    @property
    def description(self):
        if self.has_child():
            return self.child.description
        else:
            if not hasattr(self, "_description"):
                self._description = None
            return self._description

    @description.setter
    def description(self, new_desc):
        if self.has_child():
            self.child.description = new_desc
        else:
            self._description = new_desc

    @property
    def shape(self):
        if self.is_wrapper:
            return self.child.shape
        else:
            return self.native_shape

    def send(
        self,
        *location,
        inplace: bool = False,
        # local_autograd=False,
        # preinitialize_grad=False,
        no_wrap=False,
        garbage_collect_data=True,
    ):
        """Gets the pointer to a new remote object.

        One of the most commonly used methods in PySyft, this method serializes
        the object upon which it is called (self), sends the object to a remote
        worker, creates a pointer to that worker, and then returns that pointer
        from this function.

        Args:
            location: The BaseWorker object which you want to send this object
                to. Note that this is never actually the BaseWorker but instead
                a class which instantiates the BaseWorker abstraction.
            inplace: if true, return the same object instance, else a new wrapper
            # local_autograd: Use autograd system on the local machine instead of PyTorch's
                autograd on the workers.
            # preinitialize_grad: Initialize gradient for AutogradTensors to a tensor
            no_wrap: If True, wrap() is called on the created pointer
            garbage_collect_data: argument passed down to create_pointer()

        Returns:
            A tf.EagerTensor[PointerTensor] pointer to self. Note that this
            object will likely be wrapped by a tf.EagerTensor wrapper.
        """

        # If you send a pointer p1, you want the pointer to pointer p2 to control
        # the garbage collection and not the remaining old p1 (here self). Because if
        # p2 is not GCed, GCing p1 shouldn't delete the remote tensor, but if you
        # want to do so, as p2 is not GCed, you can still do `del p2`.
        # This allows to chain multiple .send().send() calls.

        if len(location) == 1:

            location = location[0]

            if hasattr(self, "child") and isinstance(self.child, PointerTensor):
                self.child.garbage_collect_data = False

            ptr = self.owner.send(self, location, garbage_collect_data=garbage_collect_data)

            ptr.description = self.description
            ptr.tags = self.tags

            # The last pointer should control remote GC, not the previous self.ptr
            if hasattr(self, "ptr") and self.ptr is not None:
                ptr_ = self.ptr()
                if ptr_ is not None:
                    ptr_.garbage_collect_data = False

            # we need to cache this weak reference to the pointer so that
            # if this method gets called multiple times we can simply re-use
            # the same pointer which was previously created
            self.ptr = weakref.ref(ptr)

            if inplace:
                self.set_()  # TODO[jason]: pretty sure this is torch specific
                self.child = ptr
                return self
            else:
                output = ptr if no_wrap else ptr.wrap()

        else:

            children = list()
            for loc in location:
                children.append(self.send(loc, no_wrap=True))

            output = syft.MultiPointerTensor(children=children)

            if not no_wrap:
                output = output.wrap()

        return output

    def get(self, *args, inplace: bool = False, **kwargs):
        """Requests the tensor/chain being pointed to, be serialized and return
            Args:
                args: args to forward to worker
                inplace: if true, return the same object instance, else a new wrapper
                kwargs: kwargs to forward to worker
            Raises:
                GetNotPermittedError: Raised if get is not permitted on this tensor
        """
        # Transfer the get() to the child attribute which is a pointer

        # if (self.has_child()):
        #     if (isinstance(self.child, syft.frameworks.torch.tensors.FixedPrecisionTensor)):
        #         if (hasattr(self.child, "child")):
        #             if (hasattr(self.child.child, "child")):
        #                 if(isinstance(self.child.child.child, syft.frameworks.torch.tensors.AdditiveSharingTensor)):
        #                     self.child.child =  self.child.child.get()
        #                     return self

        tensor = self.child.get(*args, **kwargs)

        # Clean the wrapper
        delattr(self, "child")

        if inplace:
            self.set_(tensor)  # TODO[jvmancuso]: torch-specific
            if hasattr(tensor, "child"):
                self.child = tensor.child
            return self
        else:
            return tensor

    def create_pointer(
        self,
        location: BaseWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: BaseWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
        shape=None,
    ) -> PointerTensor:
        """Creates a pointer to the "self" torch.Tensor object.

        Returns:
            A PointerTensor pointer to self. Note that this
            object will likely be wrapped by a torch.Tensor wrapper.
        """
        if id_at_location is None:
            id_at_location = self.id

        if ptr_id is None:
            if location is not None and location.id != self.owner.id:
                ptr_id = self.id
            else:
                ptr_id = syft.ID_PROVIDER.pop()

        if shape is None:
            shape = self.shape

        ptr = syft.PointerTensor.create_pointer(
            self, location, id_at_location, register, owner, ptr_id, garbage_collect_data, shape
        )

        return ptr

    def __str__(self) -> str:
        if self.has_child():
            if self.is_wrapper:
                return "(Wrapper)>" + self.child.__str__()
            else:
                return type(self).__name__ + ">" + self.child.__str__()
        else:
            return self.native___str__()

    def __repr__(self) -> str:
        if self.has_child():
            if self.is_wrapper:
                return "(Wrapper)>" + self.child.__str__()
            else:
                return type(self).__name__ + ">" + self.child.__repr__()
        else:
            out = self.native___repr__()

            big_repr = False

            if self.tags is not None and len(self.tags):
                big_repr = True
                out += "\n\tTags: "
                for tag in self.tags:
                    out += str(tag) + " "

            if self.description is not None:
                big_repr = True
                out += "\n\tDescription: " + str(self.description).split("\n")[0] + "..."

            if big_repr:
                out += "\n\tShape: " + str(self.shape)

            return out

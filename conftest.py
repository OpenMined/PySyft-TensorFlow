"""Config for tests"""
import pytest
import tensorflow as tf
import syft as sy


@pytest.fixture(scope="session", autouse=True)
def hook():
    """
    auto-hook tensorflow
    """
    return sy.TensorFlowHook(tf)


@pytest.fixture(scope="function", autouse=True)
def remote(hook):  # pylint: disable=W0621
    """
    Provide a remote worker to tests
    """
    return sy.VirtualWorker(hook, id="remote")

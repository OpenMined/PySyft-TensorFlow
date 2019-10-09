# Contributors Guidelines to PySyft-TensorFlow

## Getting Started

### Slack

A great first place to join the Community is the [Slack channel](https://slack.openmined.org).

### Issues

The open issues, including good first issues, can be found [here](https://github.com/openmined/pysyft-tensorflow/issues). This repository works with issues the same way as the [PySyft](https://github.com/OpenMined/PySyft/blob/master/CONTRIBUTING.md#issue-allocation) repository.

## Setup

### Forking a Repository

To contribute, you will need to send your pull requests via a [fork](https://help.github.com/en/articles/fork-a-repo).

### Keeping your fork up to date

If you continue to contribute, you will need to keep your fork up to date.
See this [guide](https://help.github.com/articles/syncing-a-fork/) for instructions
detailing how to sync your fork.

### Environment

We recommend creating a [Conda](https://docs.conda.io/en/latest/) environment for PySyft-TensorFlow.

```
$ conda create -n pysyft-tf python=3.7
$ source activate pysyft-tf
```

### Installing Dependencies

After forking the repository and cloning it, you can install the required dependencies

```
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

### Running Tests

We use [pytest](https://docs.pytest.org/en/latest/) to run tests.

```
$ pytest # run everything
$ pytest -k "test_fn_name_here" # run a specific failing test
```

## Documentation & Codestyle

This repository follows the same rules as [PySyft](https://github.com/OpenMined/PySyft/blob/master/CONTRIBUTING.md#documentation-and-codestyle).

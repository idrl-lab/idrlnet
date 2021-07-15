"""Define DataNode"""

import numpy
import torch
import inspect
import functools
import abc
from typing import Callable, Tuple, List, Union
from idrlnet.variable import Variables
from idrlnet.node import Node
from idrlnet.geo_utils.sympy_np import lambdify_np
from idrlnet.header import logger


class DataNode(Node):
    """A class inherits node.Node. With sampling methods implemented, the instance will generate sample points.

    :param inputs: input keys in return.
    :type inputs: Union[Tuple[str, ...], List[str]]
    :param outputs: output keys in return.
    :type outputs: Union[Tuple[str, ...], List[str]]
    :param sample_fn: Callable instances for sampling. Implementation of SampleDomain is suggested for this arg.
    :type sample_fn: Callable
    :param loss_fn: Reduce the difference between a given data and this the output of the node to a simple scalar.
                    square and L1 are implemented currently.
                    defaults to 'square'.
    :type loss_fn: str
    :param lambda_outputs: Weight for each output in return, defaults to None.
    :type lambda_outputs: Union[Tuple[str,...], List[str]]
    :param name: The name of the node.
    :type name: str
    :param sigma: The weight for the whole node. defaults to 1.
    :type sigma: float
    :param var_sigma: whether automatical loss balance technique is used. defaults to false
    :type var_sigma: bool
    :param args:
    :param kwargs:
    """

    counter = 0

    @property
    def sample_fn(self):
        return self._sample_fn

    @sample_fn.setter
    def sample_fn(self, sample_fn):
        self._sample_fn = sample_fn

    @property
    def loss_fn(self):
        return self._loss_function

    @loss_fn.setter
    def loss_fn(self, loss_fn):
        self._loss_function = loss_fn

    @property
    def lambda_outputs(self):
        return self._lambda_outputs

    @lambda_outputs.setter
    def lambda_outputs(self, lambda_outputs):
        self._lambda_outputs = lambda_outputs

    @property
    def sigma(self):
        """A weight for the domain."""
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    def sample(self) -> Variables:
        """Sample a group of points, represented by Variables.

        :return: a group of points.
        :rtype: Variables
        """
        input_vars, output_vars = self.sample_fn()
        for key, value in output_vars.items():
            if isinstance(value, torch.Tensor):
                pass
            elif isinstance(value, numpy.ndarray):
                pass
            else:
                try:
                    output_vars[key] = lambdify_np(value, input_vars)(**input_vars)
                except:
                    logger.error("unsupported constraints type.")
                    raise ValueError("unsupported constraints type.")

        try:
            return Variables({**input_vars, **output_vars}).to_torch_tensor_()
        except:
            return Variables({**input_vars, **output_vars})

    def __init__(
        self,
        inputs: Union[Tuple[str, ...], List[str]],
        outputs: Union[Tuple[str, ...], List[str]],
        sample_fn: Callable,
        loss_fn: str = "square",
        lambda_outputs: Union[Tuple[str, ...], List[str]] = None,
        name=None,
        sigma=1.0,
        var_sigma=False,
        *args,
        **kwargs,
    ):
        self.inputs: Union[Tuple, List[str]] = inputs
        self.outputs: Union[Tuple, List[str]] = outputs
        self.lambda_outputs = lambda_outputs
        if name is not None:
            self.name = name
        else:
            self.name: str = "Domain_{}".format(self.counter)
            type(self).counter += 1
        self.sigma = sigma
        self.sigma = torch.tensor(sigma, dtype=torch.float32, requires_grad=var_sigma)
        self.sample_fn: Callable = sample_fn
        self.loss_fn = loss_fn

    def __str__(self):
        str_list = [
            "DataNode properties:\n" "lambda_outputs: {}\n".format(self.lambda_outputs)
        ]
        return super().__str__() + "".join(str_list)


def get_data_node(
    fun: Callable,
    name=None,
    loss_fn="square",
    sigma=1.0,
    var_sigma=False,
    *args,
    **kwargs,
) -> DataNode:
    """Construct a datanode from sampling functions.

    :param fun: Each call of the Callable object should return a sampling dict.
    :type fun: Callable
    :param name: name of the generated Datanode, defaults to None
    :type name: str
    :param loss_fn: Specify a loss function for the data node.
    :type loss_fn: str
    :param args:
    :param kwargs:
    :return: An instance of Datanode
    :rtype: DataNode
    """
    in_, out_ = fun()
    inputs = list(in_.keys())
    outputs = list(out_.keys())
    lambda_outputs = list(filter(lambda x: x.startswith("lambda_"), outputs))
    outputs = list(filter(lambda x: not x.startswith("lambda_"), outputs))
    name = (
        (fun.__name__ if inspect.isfunction(fun) else type(fun).__name__)
        if name is None
        else name
    )
    dn = DataNode(
        inputs=inputs,
        outputs=outputs,
        sample_fn=fun,
        lambda_outputs=lambda_outputs,
        loss_fn=loss_fn,
        name=name,
        sigma=sigma,
        var_sigma=var_sigma,
        *args,
        **kwargs,
    )
    return dn


def datanode(
    _fun: Callable = None,
    name=None,
    loss_fn="square",
    sigma=1.0,
    var_sigma=False,
    **kwargs,
):
    """As an alternative, decorate Callable classes as Datanode."""

    def wrap(fun):
        if inspect.isclass(fun):
            assert issubclass(
                fun, SampleDomain
            ), f"{fun} should be subclass of .data.Sample"
            fun = fun()
        assert isinstance(fun, Callable)

        @functools.wraps(fun)
        def wrapped_fun():
            dn = get_data_node(
                fun,
                name=name,
                loss_fn=loss_fn,
                sigma=sigma,
                var_sigma=var_sigma,
                **kwargs,
            )
            return dn

        return wrapped_fun

    return wrap if _fun is None else wrap(_fun)


def get_data_nodes(funs: List[Callable], *args, **kwargs) -> Tuple[DataNode]:
    if "names" in kwargs:
        names = kwargs.pop("names")
        return tuple(
            get_data_node(fun, name=name, *args, **kwargs)
            for fun, name in zip(funs, names)
        )
    else:
        return tuple(get_data_node(fun, *args, **kwargs) for fun in funs)


class SampleDomain(metaclass=abc.ABCMeta):
    """Template for Callable sampling function."""

    @abc.abstractmethod
    def sampling(self, *args, **kwargs):
        """The method returns sampling points"""
        raise NotImplementedError(f"{type(self)}.sampling method not implemented")

    def __call__(self, *args, **kwargs):
        return self.sampling(self, *args, **kwargs)

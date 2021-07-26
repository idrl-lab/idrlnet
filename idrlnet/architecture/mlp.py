"""This module provide some MLP architectures."""

import torch
import math
from collections import OrderedDict
from idrlnet.architecture.layer import (
    get_linear_layer,
    get_activation_layer,
    Initializer,
    Activation,
)
from typing import List, Union, Tuple
from idrlnet.header import logger
from idrlnet.net import NetNode
import enum


class MLP(torch.nn.Module):
    """A subclass of torch.nn.Module customizes a multiple linear perceptron network.

    :param n_seq: Define neuron numbers in each layer. The number of the first and the last should be in
                  keeping with inputs and outputs.
    :type n_seq: List[int]
    :param activation: By default, the activation is `Activation.swish`.
    :type activation: Union[Activation,List[Activation]]
    :param initialization:
    :type initialization:Initializer
    :param weight_norm: If weight normalization is used.
    :type weight_norm: bool
    :param name: Symbols will appear in the name of each layer. Do not confuse with the netnode name.
    :type name: str
    :param args:
    :param kwargs:
    """

    def __init__(
        self,
        n_seq: List[int],
        activation: Union[Activation, List[Activation]] = Activation.swish,
        initialization: Initializer = Initializer.kaiming_uniform,
        weight_norm: bool = True,
        name: str = "mlp",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.layers = OrderedDict()
        current_activation = ""
        assert isinstance(n_seq, Activation) or isinstance(n_seq, list)
        for i in range(len(n_seq) - 1):
            if isinstance(activation, list):
                current_activation = activation[i]
            elif i < len(n_seq) - 2:
                current_activation = activation
            self.layers["{}_{}".format(name, i)] = get_linear_layer(
                n_seq[i], n_seq[i + 1], weight_norm, initialization, *args, **kwargs
            )
            if (
                isinstance(activation, Activation) and i < len(n_seq) - 2
            ) or isinstance(activation, list):
                if current_activation == "none":
                    continue
                self.layers["{}_{}_activation".format(name, i)] = get_activation_layer(
                    current_activation, *args, **kwargs
                )
        self.layers = torch.nn.ModuleDict(self.layers)

    def forward(self, x):
        n_layers = len(self.layers)
        i = 0
        for name, layer in self.layers.items():
            x = layer(x)
            if i == n_layers - 1:
                break
            i += 1
        return x


class Siren(torch.nn.Module):
    def __init__(
        self,
        n_seq: List[int],
        first_omega: float = 30.0,
        omega: float = 30.0,
        name: str = "siren",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.layers = OrderedDict()
        self.first_omega = first_omega
        self.omega = omega
        assert isinstance(n_seq, str) or isinstance(n_seq, list)
        for i in range(len(n_seq) - 1):
            if i == 0:
                self.layers["{}_{}".format(name, i)] = self.get_siren_layer(
                    n_seq[i], n_seq[i + 1], True, first_omega
                )
            else:
                self.layers["{}_{}".format(name, i)] = self.get_siren_layer(
                    n_seq[i], n_seq[i + 1], False, omega
                )
            if i < (len(n_seq) - 2):
                self.layers["{}_{}_activation".format(name, i)] = get_activation_layer(
                    Activation.sin, *args, **kwargs
                )

        self.layers = torch.nn.ModuleDict(self.layers)

    @staticmethod
    def get_siren_layer(
        input_dim: int, output_dim: int, is_first: bool, omega_0: float
    ):
        layer = torch.nn.Linear(input_dim, output_dim)
        dim = input_dim
        if is_first:
            torch.nn.init.uniform_(layer.weight.data, -1.0 / dim, 1.0 / dim)
        else:
            torch.nn.init.uniform_(
                layer.weight.data,
                -1.0 * math.sqrt(6.0 / dim) / omega_0,
                math.sqrt(6.0 / dim) / omega_0,
            )
        torch.nn.init.uniform_(
            layer.bias.data, -1 * math.sqrt(1 / dim), math.sqrt(1 / dim)
        )
        return layer

    def forward(self, x):
        i = 0
        n_layers = len(self.layers)
        for name, layer in self.layers.items():
            x = layer(x)
            if isinstance(layer, torch.nn.Linear) and i < n_layers - 1:
                x = self.first_omega * x if i == 0 else self.omega * x
            i += 1
        return x


class SingleVar(torch.nn.Module):
    """Wrapper a single parameter to represent an unknown coefficient in inverse problem.

    :param initialization: initialization value for the parameter. The default is 0.01
    :type initialization: float
    """

    def __init__(self, initialization: float = 1.0):
        super().__init__()
        self.value = torch.nn.Parameter(torch.Tensor([initialization]))

    def forward(self, x) -> torch.Tensor:
        return x[:, :1] * 0.0 + self.value

    def get_value(self) -> torch.Tensor:
        return self.value


class BoundedSingleVar(torch.nn.Module):
    """Wrapper a single parameter to represent an unknown coefficient in inverse problem with the upper and lower bound.

    :param lower_bound: The lower bound for the parameter.
    :type lower_bound: float
    :param upper_bound: The upper bound for the parameter.
    :type upper_bound: float
    """

    def __init__(self, lower_bound, upper_bound):
        super().__init__()
        self.value = torch.nn.Parameter(torch.Tensor([0.0]))
        self.layer = torch.nn.Sigmoid()
        self.ub, self.lb = upper_bound, lower_bound

    def forward(self, x) -> torch.Tensor:
        return x[:, :1] * 0.0 + self.layer(self.value) * (self.ub - self.lb) + self.lb

    def get_value(self) -> torch.Tensor:
        return self.layer(self.value) * (self.ub - self.lb) + self.lb


class Arch(enum.Enum):
    """Enumerate pre-defined neural networks."""

    mlp = "mlp"
    toy = "toy"
    mlp_xl = "mlp_xl"
    single_var = "single_var"
    bounded_single_var = "bounded_single_var"
    siren = "siren"


def get_net_node(
    inputs: Union[Tuple[str, ...], List[str]],
    outputs: Union[Tuple[str, ...], List[str]],
    arch: Arch = None,
    name=None,
    *args,
    **kwargs,
) -> NetNode:
    """Get a net node wrapping networks with pre-defined configurations

    :param inputs: Input symbols for the generated node.
    :type inputs: Union[Tuple[str, ...]
    :param outputs: Output symbols for the generated node.
    :type outputs: Union[Tuple[str, ...]
    :param arch: One can choose one of
                 - Arch.mlp
                 - Arch.mlp_xl(more layers and more neurons)
                 - Arch.single_var
                 - Arch.bounded_single_var
    :type arch: Arch
    :param name: The name of the generated node.
    :type name: str
    :param args:
    :param kwargs:
    :return:
    """
    arch = Arch.mlp if arch is None else arch
    if "evaluate" in kwargs.keys():
        evaluate = kwargs.pop("evaluate")
    else:
        if arch == Arch.mlp:
            seq = (
                kwargs["seq"]
                if "seq" in kwargs.keys()
                else [len(inputs), 20, 20, 20, 20, len(outputs)]
            )
            evaluate = MLP(
                n_seq=seq,
                activation=Activation.swish,
                initialization=Initializer.kaiming_uniform,
                weight_norm=True,
            )
        elif arch == Arch.toy:
            evaluate = SimpleExpr("nothing")
        elif arch == Arch.mlp_xl or arch == "fc":
            seq = (
                kwargs["seq"]
                if "seq" in kwargs.keys()
                else [len(inputs), 512, 512, 512, 512, 512, 512, len(outputs)]
            )
            evaluate = MLP(
                n_seq=seq,
                activation=Activation.silu,
                initialization=Initializer.kaiming_uniform,
                weight_norm=True,
            )
        elif arch == Arch.single_var:
            evaluate = SingleVar(initialization=kwargs.get("initialization", 1.0))
        elif arch == Arch.bounded_single_var:
            evaluate = BoundedSingleVar(
                lower_bound=kwargs["lower_bound"], upper_bound=kwargs["upper_bound"]
            )
        elif arch == Arch.siren:
            seq = (
                kwargs["seq"]
                if "seq" in kwargs.keys()
                else [len(inputs), 512, 512, 512, 512, 512, 512, len(outputs)]
            )
            evaluate = Siren(n_seq=seq)
        else:
            logger.error(f"{arch} is not supported!")
            raise NotImplementedError(f"{arch} is not supported!")
    nn = NetNode(
        inputs=inputs, outputs=outputs, net=evaluate, name=name, *args, **kwargs
    )
    return nn


def get_shared_net_node(
    shared_node: NetNode,
    inputs: Union[Tuple[str, ...], List[str]],
    outputs: Union[Tuple[str, ...], List[str]],
    name=None,
    *args,
    **kwargs,
) -> NetNode:
    """Construct a netnode, the net of which is shared by a given netnode. One can specify different inputs and outputs
    just like an independent netnode. However, the net parameters may have multiple references. Thus the step
    operations during optimization should only be applied once.

    :param shared_node: An existing netnode, the network of which will be shared.
    :type shared_node: NetNode
    :param inputs: Input symbols for the generated node.
    :type inputs: Union[Tuple[str, ...]
    :param outputs: Output symbols for the generated node.
    :type outputs: Union[Tuple[str, ...]
    :param name: The name of the generated node.
    :type name: str
    :param args:
    :param kwargs:
    :return:
    """
    nn = NetNode(
        inputs, outputs, shared_node.net, is_reference=True, name=name, *args, **kwargs
    )
    return nn


def get_inter_name(length: int, prefix: str):
    return [prefix + f"_{i}" for i in range(length)]


class SimpleExpr(torch.nn.Module):
    """This class is for testing. One can override SimpleExper.forward to represent complex formulas."""

    def __init__(self, expr, name="expr"):
        super().__init__()
        self.evaluate = expr
        self.name = name
        self._placeholder = torch.nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        return (
            self._placeholder
            + x[:, :1] * x[:, :1] / 2
            + x[:, 1:] * x[:, 1:] / 2
            - self._placeholder
        )

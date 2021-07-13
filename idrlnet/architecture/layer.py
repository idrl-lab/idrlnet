""" The module provide elements for construct MLP."""

import enum
import math
import torch
from idrlnet.header import logger

__all__ = ["Activation", "Initializer", "get_activation_layer", "get_linear_layer"]


class Activation(enum.Enum):
    relu = "relu"
    silu = "silu"
    selu = "selu"
    sigmoid = "sigmoid"
    tanh = "tanh"
    swish = "swish"
    poly = "poly"
    sin = "sin"
    leaky_relu = "leaky_relu"


class Initializer(enum.Enum):
    Xavier_uniform = "Xavier_uniform"
    constant = "constant"
    kaiming_uniform = "kaiming_uniform"
    default = "default"


def get_linear_layer(
    input_dim: int,
    output_dim: int,
    weight_norm=False,
    initializer: Initializer = Initializer.Xavier_uniform,
    *args,
    **kwargs,
):
    layer = torch.nn.Linear(input_dim, output_dim)
    init_method = InitializerFactory.get_initializer(initializer=initializer, **kwargs)
    init_method(layer.weight)
    torch.nn.init.constant_(layer.bias, 0.0)
    if weight_norm:
        layer = torch.nn.utils.weight_norm(layer)
    return layer


def get_activation_layer(activation: Activation = Activation.swish, *args, **kwargs):
    return ActivationFactory.get_from_string(activation)


def modularize(fun_generator):
    def wrapper(fun):
        class _LambdaModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fun = fun_generator(fun)

            def forward(self, x):
                # x = self.fun(-x)
                x = self.fun(x)
                return x

        return type(fun.name, (_LambdaModule,), {})()

    return wrapper


class ActivationFactory:
    @staticmethod
    @modularize
    def get_from_string(activation: Activation, *args, **kwargs):
        if activation == Activation.relu:
            return torch.relu
        elif activation == Activation.selu:
            return torch.selu
        elif activation == Activation.sigmoid:
            return torch.sigmoid
        elif activation == Activation.tanh:
            return torch.tanh
        elif activation == Activation.swish:
            return swish
        elif activation == Activation.poly:
            return poly
        elif activation == Activation.sin:
            return torch.sin
        elif activation == Activation.silu:
            return Silu()
        else:
            logger.error(f"Activation {activation} is not supported!")
            raise NotImplementedError(
                "Activation " + activation.name + " is not supported"
            )


class Silu:
    def __init__(self):
        try:
            self.m = torch.nn.SiLU()
        except:
            self.m = lambda x: x * torch.sigmoid(x)

    def __call__(self, x):
        return self.m(x)


def leaky_relu(x, leak=0.1):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def triangle_wave(x):
    y = 0.0
    for i in range(3):
        y += (
            (-1.0) ** (i)
            * torch.sin(2.0 * math.pi * (2.0 * i + 1.0) * x)
            / (2.0 * i + 1.0) ** (2)
        )
    y = 0.5 * (8 / (math.pi ** 2) * y) + 0.5
    return y


def swish(x):
    return x * torch.sigmoid(x)


def hard_swish(x):
    return x * torch.sigmoid(100.0 * x)


def poly(x):
    axis = len(x.get_shape()) - 1
    return torch.cat([x ** 3, x ** 2, x], axis)


def fourier(x, terms=10):
    axis = len(x.get_shape()) - 1
    x_list = []
    for i in range(terms):
        x_list.append(torch.sin(2 * math.pi * i * x))
        x_list.append(torch.cos(2 * math.pi * i * x))
    return torch.cat(x_list, axis)


class InitializerFactory:
    @staticmethod
    def get_initializer(initializer: Initializer, *args, **kwargs):
        # todo: more
        if initializer == Initializer.Xavier_uniform:
            return torch.nn.init.xavier_uniform_
        elif initializer == Initializer.constant:
            return lambda x: torch.nn.init.constant_(x, kwargs["constant"])
        elif initializer == Initializer.kaiming_uniform:
            return lambda x: torch.nn.init.kaiming_uniform_(
                x, mode="fan_in", nonlinearity="relu"
            )
        elif initializer == Initializer.default:
            return lambda x: x
        else:
            logger.error("initialization " + initializer.name + " is not supported")
            raise NotImplementedError(
                "initialization " + initializer.name + " is not supported"
            )

"""Define Basic Node"""
from typing import Callable, List

from idrlnet.torch_util import torch_lambdify
from idrlnet.variable import Variables
from idrlnet.header import DIFF_SYMBOL

__all__ = ["Node"]


class Node(object):
    @property
    def inputs(self) -> List[str]:
        try:
            return self._inputs
        except:
            self._inputs = tuple()
            return self._inputs

    @inputs.setter
    def inputs(self, inputs: List[str]):
        self._inputs = inputs

    @property
    def outputs(self) -> List[str]:
        try:
            return self._outputs
        except:
            self._outputs = tuple()
            return self._outputs

    @outputs.setter
    def outputs(self, outputs: List[str]):
        self._outputs = outputs

    @property
    def derivatives(self) -> List[str]:
        try:
            return self._derivatives
        except:
            self._derivatives = []
            return self._derivatives

    @derivatives.setter
    def derivatives(self, derivatives: List[str]):
        self._derivatives = derivatives

    @property
    def evaluate(self) -> Callable:
        return self._evaluate

    @evaluate.setter
    def evaluate(self, evaluate: Callable):
        self._evaluate = evaluate

    @property
    def name(self) -> str:
        try:
            return self._name
        except:
            self._name = "Node" + str(id(self))
            return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @classmethod
    def new_node(
        cls,
        name: str = None,
        tf_eq: Callable = None,
        free_symbols: List[str] = None,
        *args,
        **kwargs
    ) -> "Node":
        node = cls()
        node.evaluate = LambdaTorchFun(free_symbols, tf_eq, name)
        node.inputs = [x for x in free_symbols if DIFF_SYMBOL not in x]
        node.derivatives = [x for x in free_symbols if DIFF_SYMBOL in x]
        node.outputs = [
            name,
        ]
        node.name = name
        return node

    def __str__(self):
        str_list = [
            "Basic properties:\n",
            "name: {}\n".format(self.name),
            "inputs: {}\n".format(self.inputs),
            "derivatives: {}\n".format(self.derivatives),
            "outputs: {}\n".format(self.outputs),
        ]
        return "".join(str_list)


class LambdaTorchFun:
    def __init__(self, free_symbols, tf_eq, name):
        self.lambda_tf_eq = torch_lambdify(free_symbols, tf_eq)
        self.tf_eq = tf_eq
        self.name = name
        self.free_symbols = free_symbols

    def __call__(self, var: Variables):
        new_var = {}
        for key, values in var.items():
            new_var[key] = values
        return {self.name: self.lambda_tf_eq(**new_var)}

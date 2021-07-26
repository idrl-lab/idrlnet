"""Define NetNode"""
import torch
from idrlnet.node import Node
from typing import Tuple, List, Dict, Union
from contextlib import ExitStack

__all__ = ["NetNode"]


class WrapEvaluate:
    def __init__(self, binding_node: "NetNode"):
        self.binding_node = binding_node

    def __call__(self, inputs):
        keep_type = None
        if isinstance(inputs, dict):
            keep_type = dict
            inputs = torch.cat(
                [
                    torch.tensor(inputs[key], dtype=torch.float32)
                    if not isinstance(inputs[key], torch.Tensor)
                    else inputs[key]
                    for key in inputs
                ],
                dim=1,
            )
        with ExitStack() as es:
            if self.binding_node.require_no_grad:
                es.enter_context(torch.no_grad())
            output_var = self.binding_node.net(inputs)
        if keep_type == dict:
            output_var = {
                outkey: output_var[:, i : i + 1]
                for i, outkey in enumerate(self.binding_node.outputs)
            }
        return output_var


class NetNode(Node):
    counter = 0

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, fixed: bool):
        self._fixed = fixed

    @property
    def require_no_grad(self):
        return self._require_no_grad

    @require_no_grad.setter
    def require_no_grad(self, require_no_grad: bool):
        self._require_no_grad = require_no_grad

    @property
    def is_reference(self):
        return self._is_reference

    @is_reference.setter
    def is_reference(self, is_reference: bool):
        self._is_reference = is_reference

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, net):
        self._net = net

    def __init__(
        self,
        inputs: Union[Tuple, List[str]],
        outputs: Union[Tuple, List[str]],
        net: torch.nn.Module,
        fixed: bool = False,
        require_no_grad: bool = False,
        is_reference=False,
        name=None,
        *args,
        **kwargs
    ):
        self.is_reference = is_reference
        self.inputs: Union[Tuple, List[str]] = inputs
        self.outputs: Union[Tuple, List[str]] = outputs
        self.derivatives: Union[Tuple, List[str]] = []
        self.net: torch.nn.Module = net
        self.require_no_grad = require_no_grad
        self.fixed = fixed
        if name is not None:
            self.name = name
        else:
            # todo: make sure this is working
            self.name: str = "net_{}".format(type(self).counter)
            type(self).counter += 1
        self.evaluate = WrapEvaluate(binding_node=self)

    def __str__(self):
        basic_info = super().__str__()

        return basic_info + str(self.net)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        return self.net.load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        return self.net.state_dict(destination, prefix, keep_vars)

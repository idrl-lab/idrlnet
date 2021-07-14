"""Define PdeNode"""

from typing import List, Dict
from idrlnet.node import Node
from idrlnet.torch_util import _replace_derivatives
from idrlnet.header import DIFF_SYMBOL
from idrlnet.variable import Variables

__all__ = ["PdeNode", "ExpressionNode"]


class PdeEvaluate:
    """A wrapper for PdeNode.evaluate"""

    def __init__(self, binding_pde):
        self.binding_pde = binding_pde

    def __call__(self, inputs: Variables) -> Variables:
        result = Variables()
        for node in self.binding_pde.sub_nodes:
            sub_inputs = {
                k: v
                for k, v in Variables(inputs).items()
                if k in node.inputs or k in node.derivatives
            }
            r = node.evaluate(sub_inputs)
            result.update(r)
        return result


class PdeNode(Node):
    @property
    def suffix(self) -> str:
        return self._suffix

    @suffix.setter
    def suffix(self, suffix: str):
        # todo: check suffix
        self._suffix = suffix

    @property
    def equations(self) -> Dict:
        return self._equations

    @equations.setter
    def equations(self, equations: Dict):
        self._equations = equations

    @property
    def sub_nodes(self) -> List:
        return self._sub_nodes

    @sub_nodes.setter
    def sub_nodes(self, sub_nodes: List):
        self._sub_nodes = sub_nodes

    def __init__(self, suffix: str = "", **kwargs):
        if len(suffix) > 0:
            self.suffix = "[" + kwargs["suffix"] + "]"  # todo: check prefix
        else:
            self.suffix = ""
        self.name = type(self).__name__ + self.suffix
        self.evaluate = PdeEvaluate(self)

    def make_nodes(self) -> None:
        self.sub_nodes = []
        free_symbols_set = set()
        name_set = set()
        for name, eq in self.equations.items():
            torch_eq = _replace_derivatives(eq)
            free_symbols = [x.name for x in torch_eq.free_symbols]
            free_symbols_set.update(set(free_symbols))
            name = name + self.suffix
            node = Node.new_node(name, torch_eq, free_symbols)
            name_set.update({name})
            self.sub_nodes.append(node)
        self.inputs = [x for x in free_symbols_set if DIFF_SYMBOL not in x]
        self.derivatives = [x for x in free_symbols_set if DIFF_SYMBOL in x]
        self.outputs = [x for x in name_set]

    def __str__(self):
        subnode_str = "\n\n".join(
            str(sub_node) + "Equation: \n" + str(self.equations[sub_node.name])
            for sub_node in self.sub_nodes
        )
        return super().__str__() + "subnodes".center(30, "-") + "\n" + subnode_str


# todo: test required
class ExpressionNode(PdeNode):
    def __init__(self, expression, name, **kwargs):
        super().__init__(**kwargs)
        self.equations = dict()
        self.equations[name] = expression
        self.name = name
        self.make_nodes()

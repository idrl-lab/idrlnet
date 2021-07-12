"""Operators in PDE

"""
import numpy as np
import sympy as sp
import torch
from idrlnet.node import Node
from idrlnet.pde import PdeNode
from sympy import Symbol, Function, symbols, Number
from typing import Union, List
from idrlnet.torch_util import integral, _replace_derivatives, torch_lambdify
from idrlnet.variable import Variables

__all__ = [
    "NormalGradient",
    "Difference",
    "Derivative",
    "Curl",
    "Divergence",
    "ICNode",
    "Int1DNode",
    "IntEq",
]


class NormalGradient(PdeNode):
    def __init__(self, T: Union[str, Symbol, float, int], dim=3, time=True):
        super().__init__()
        self.T = T
        self.dim = dim
        self.time = time

        x, y, z, normal_x, normal_y, normal_z, t = symbols(
            "x y z normal_x normal_y normal_z t"
        )

        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        T = Function(T)(*input_variables)

        self.equations = {
            "normal_gradient_"
            + self.T: (
                normal_x * T.diff(x) + normal_y * T.diff(y) + normal_z * T.diff(z)
            )
        }
        self.make_nodes()


class Difference(PdeNode):
    def __init__(
        self,
        T: Union[str, Symbol, float, int],
        S: Union[str, Symbol, float, int],
        dim=3,
        time=True,
    ):
        super().__init__()
        self.T = T
        self.S = S
        self.dim = dim
        self.time = time
        x, y, z = symbols("x y z")
        t = Symbol("t")
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # variables to set the gradients (example Temperature)
        T = Function(T)(*input_variables)
        S = Function(S)(*input_variables)

        # set equations
        self.equations = {}
        self.equations["difference_" + self.T + "_" + self.S] = T - S
        self.make_nodes()


class Derivative(PdeNode):
    def __init__(
        self,
        T: Union[str, Symbol, float, int],
        p: Union[str, Symbol],
        S: Union[str, Symbol, float, int] = 0.0,
        dim=3,
        time=True,
    ):
        super().__init__()
        self.T = T
        self.S = S
        self.dim = dim
        self.time = time
        x, y, z = symbols("x y z")
        t = Symbol("t")

        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")
        if type(S) is str:
            S = Function(S)(*input_variables)
        elif type(S) in [float, int]:
            S = Number(S)
        if isinstance(p, str):
            p = Symbol(p)
        T = Function(T)(*input_variables)
        self.equations = {}
        if isinstance(S, Function):
            self.equations[
                "derivative_" + self.T + ":" + str(p) + "_" + str(self.S)
            ] = (T.diff(p) - S)
        else:
            self.equations["derivative_" + self.T + ":" + str(p)] = T.diff(p) - S
        self.make_nodes()


class Curl(PdeNode):
    def __init__(self, vector, curl_name=None):
        super().__init__()
        if curl_name is None:
            curl_name = ["u", "v", "w"]
        x, y, z = symbols("x y z")
        input_variables = {"x": x, "y": y, "z": z}

        v_0 = vector[0]
        v_1 = vector[1]
        v_2 = vector[2]
        if type(v_0) is str:
            v_0 = Function(v_0)(*input_variables)
        elif type(v_0) in [float, int]:
            v_0 = Number(v_0)
        if type(v_1) is str:
            v_1 = Function(v_1)(*input_variables)
        elif type(v_1) in [float, int]:
            v_1 = Number(v_1)
        if type(v_2) is str:
            v_2 = Function(v_2)(*input_variables)
        elif type(v_2) in [float, int]:
            v_2 = Number(v_2)

        curl_0 = v_2.diff(y) - v_1.diff(z)
        curl_1 = v_0.diff(z) - v_2.diff(x)
        curl_2 = v_1.diff(x) - v_0.diff(y)

        self.equations = {}
        self.equations[curl_name[0]] = curl_0
        self.equations[curl_name[1]] = curl_1
        self.equations[curl_name[2]] = curl_2


class Divergence(PdeNode):
    def __init__(self, vector, div_name="div_v"):
        super().__init__()
        x, y, z = symbols("x y z")

        input_variables = {"x": x, "y": y, "z": z}

        v_0 = vector[0]
        v_1 = vector[1]
        v_2 = vector[2]

        if type(v_0) is str:
            v_0 = Function(v_0)(*input_variables)
        elif type(v_0) in [float, int]:
            v_0 = Number(v_0)
        if type(v_1) is str:
            v_1 = Function(v_1)(*input_variables)
        elif type(v_1) in [float, int]:
            v_1 = Number(v_1)
        if type(v_2) is str:
            v_2 = Function(v_2)(*input_variables)
        elif type(v_2) in [float, int]:
            v_2 = Number(v_2)

        self.equations = {}
        self.equations[div_name] = v_0 + v_1 + v_2


class ICNode(PdeNode):
    def __init__(
        self,
        T: Union[str, Symbol, int, float, List[Union[str, Symbol, int, float]]],
        dim: int = 2,
        time: bool = False,
        reduce_name: str = None,
    ):
        super().__init__()
        if reduce_name is None:
            reduce_name = str(T)
        self.T = T
        self.dim = dim
        self.time = time
        self.reduce_name = reduce_name

        x, y, z = symbols("x y z")
        normal_x = Symbol("normal_x")
        normal_y = Symbol("normal_y")
        normal_z = Symbol("normal_z")
        area = Symbol("area")

        t = Symbol("t")

        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        def sympify_T(
            T: Union[str, Symbol, int, float, List[Union[str, Symbol, int, float]]]
        ) -> Union[Symbol, List[Symbol]]:
            if isinstance(T, list):
                return [sympify_T(_T) for _T in T]
            elif type(T) is str:
                T = Function(T)(*input_variables)
            elif type(T) in [float, int]:
                T = Number(T)
            return T

        T = sympify_T(T)
        # set equations
        self.equations = {}
        if isinstance(T, list):
            if self.dim == 3:
                self.equations["integral_" + self.reduce_name] = integral(
                    (normal_x * T[0] + normal_y * T[1] + normal_z * T[2]) * area
                )
            if self.dim == 2:
                self.equations["integral_" + self.reduce_name] = integral(
                    (normal_x * T[0] + normal_y * T[1]) * area
                )
        else:
            self.equations["integral_" + self.reduce_name] = integral(T * area)
        self.make_nodes()


class Int1DNode(PdeNode):
    counter = 0

    def __init__(
        self,
        expression,
        expression_name,
        lb,
        ub,
        var: Union[str, sp.Symbol] = "s",
        degree=20,
        **kwargs
    ):
        super().__init__(**kwargs)
        x = sp.Symbol("x")
        self.equations = {}
        self.var = sp.Symbol(var) if isinstance(var, str) else var
        self.degree = degree
        quad_s, quad_w = np.polynomial.legendre.leggauss(self.degree)
        self.quad_s = torch.tensor(quad_s, dtype=torch.float32)
        self.quad_w = torch.tensor(quad_w, dtype=torch.float32)

        if type(lb) is str:
            self.lb = sp.Function(lb)(x)
        elif type(lb) in [float, int]:
            self.lb = sp.Number(lb)
        else:
            self.lb = lb

        if type(ub) is str:
            self.ub = sp.Function(ub)(x)
        elif type(ub) in [float, int]:
            self.ub = sp.Number(ub)
        else:
            self.ub = ub

        if type(expression) in [float, int]:
            self.equations[expression_name] = sp.Number(expression)
        elif isinstance(expression, sp.Expr):
            self.equations[expression_name] = expression
        else:
            raise

        if "funs" in kwargs.keys():
            self.funs = kwargs["funs"]
        else:
            self.funs = {}
        self.computable_name = set(
            *[fun["output_map"].values() for _, fun in self.funs.items()]
        )
        self.fun_require_input = set(
            *[
                set(fun["eval"].inputs) - set(fun["input_map"].keys())
                for _, fun in self.funs.items()
            ]
        )

        self.make_nodes()

    def make_nodes(self) -> None:
        self.sub_nodes = []
        free_symbols_set = set()
        name_set = set()
        for name, eq in self.equations.items():
            self.lb = _replace_derivatives(self.lb)
            self.ub = _replace_derivatives(self.ub)
            eq = _replace_derivatives(eq)
            free_symbols_set.update(set(x.name for x in self.ub.free_symbols))
            free_symbols_set.update(set(x.name for x in self.lb.free_symbols))
            free_symbols_set.update(set(x.name for x in eq.free_symbols))
            for ele in self.fun_require_input:
                free_symbols_set.add(ele)
            if self.var.name in free_symbols_set:
                free_symbols_set.remove(self.var.name)

            name = name + self.suffix
            node = self.new_node(name, eq, list(free_symbols_set))
            name_set.update({name})
            self.sub_nodes.append(node)

        self.inputs = [x for x in free_symbols_set if x not in self.funs.keys()]
        self.derivatives = []
        self.outputs = [x for x in name_set]

    def new_node(
        self,
        name: str = None,
        tf_eq: sp.Expr = None,
        free_symbols: List[str] = None,
        *args,
        **kwargs
    ):
        out_symbols = [x for x in free_symbols if x not in self.funs.keys()]
        lb_lambda = torch_lambdify(out_symbols, self.lb)
        ub_lambda = torch_lambdify(out_symbols, self.ub)
        eq_lambda = torch_lambdify([*free_symbols, self.var.name], tf_eq)
        node = Node()
        node.evaluate = IntEq(
            self, lb_lambda, ub_lambda, out_symbols, free_symbols, eq_lambda, name
        )
        node.inputs = [x for x in free_symbols if x not in self.funs.keys()]
        node.derivatives = []
        node.outputs = [name]
        node.name = name
        return node


class IntEq:
    def __init__(
        self,
        binding_node,
        lb_lambda,
        ub_lambda,
        out_symbols,
        free_symbols,
        eq_lambda,
        name,
    ):
        self.binding_node = binding_node
        self.lb_lambda = lb_lambda
        self.ub_lambda = ub_lambda
        self.out_symbols = out_symbols
        self.free_symbols = free_symbols
        self.eq_lambda = eq_lambda
        self.name = name

    def __call__(self, var: Variables):
        var = {k: v for k, v in var.items()}
        lb_value = self.lb_lambda(
            **{k: v for k, v in var.items() if k in self.out_symbols}
        )
        ub_value = self.ub_lambda(
            **{k: v for k, v in var.items() if k in self.out_symbols}
        )

        xx = dict()
        for syp in self.free_symbols:
            if syp not in var.keys():
                continue
            value = var[syp]
            _value = torch.ones_like(self.binding_node.quad_s) * value
            _value = _value.reshape(-1, 1)
            xx.update({syp: _value})

        quad_w = (ub_value - lb_value) / 2 * self.binding_node.quad_w
        quad_s = (self.binding_node.quad_s + 1) * (ub_value - lb_value) / 2 + lb_value
        shape = quad_w.shape

        quad_w = quad_w.reshape(-1, 1)
        quad_s = quad_s.reshape(-1, 1)

        new_var = dict()
        for _, fun in self.binding_node.funs.items():
            input_map = fun["input_map"]
            output_map = fun["output_map"]
            tmp_var = dict()
            for k, v in xx.items():
                tmp_var[k] = v
            for k, v in input_map.items():
                tmp_var[k] = quad_s
            res = fun["eval"].evaluate(tmp_var)
            for k, v in output_map.items():
                res[v] = res.pop(k)
            new_var.update(res)
        xx.update(new_var)

        values = quad_w * self.eq_lambda(
            **dict(**{self.binding_node.var.name: quad_s}, **xx)
        )
        values = values.reshape(shape)
        return {self.name: values.sum(1, keepdim=True)}

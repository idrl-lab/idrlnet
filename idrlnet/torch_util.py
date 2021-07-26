"""
conversion utils for sympy expression and torch functions.
todo: replace sampling method in GEOMETRY
"""

from sympy import lambdify, Symbol, Derivative, Function, Basic
from sympy.utilities.lambdify import implemented_function
from sympy.printing.str import StrPrinter
import torch
from idrlnet.header import DIFF_SYMBOL
from functools import reduce

__all__ = ["integral", "torch_lambdify"]


def integral_fun(x):
    if isinstance(x, torch.Tensor):
        return torch.sum(input=x, dim=0, keepdim=True) * torch.ones_like(x)
    return x


integral = implemented_function("integral", lambda x: integral_fun(x))


def torch_lambdify(r, f, *args, **kwargs):
    try:
        f = float(f)
    except:
        pass
    if isinstance(f, (float, int, bool)):  # constant function

        def loop_lambda(constant):
            return lambda **x: torch.zeros_like(next(iter(x.items()))[1]) + constant

        lambdify_f = loop_lambda(f)
    else:
        lambdify_f = lambdify([k for k in r], f, [TORCH_SYMPY_PRINTER], *args, **kwargs)
        # lambdify_f = lambdify([k for k in r], f, *args, **kwargs)
    return lambdify_f


# todo: more functions
TORCH_SYMPY_PRINTER = {
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "exp": torch.exp,
    "sqrt": torch.sqrt,
    "Abs": torch.abs,
    "tanh": torch.tanh,
    "DiracDelta": torch.zeros_like,
    "Heaviside": lambda x: torch.heaviside(x, torch.tensor([0.0])),
    "amin": lambda x: reduce(lambda y, z: torch.minimum(y, z), x),
    "amax": lambda x: reduce(lambda y, z: torch.maximum(y, z), x),
    "Min": lambda *x: reduce(lambda y, z: torch.minimum(y, z), x),
    "Max": lambda *x: reduce(lambda y, z: torch.maximum(y, z), x),
    "equal": lambda x, y: torch.isclose(x, y),
    "Xor": torch.logical_xor,
    "log": torch.log,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "asin": torch.arcsin,
    "acos": torch.arccos,
    "atan": torch.arctan,
}


def _reduce_sum(x: torch.Tensor):
    return torch.sum(x, dim=0, keepdim=True)


def _replace_derivatives(expr):
    while len(expr.atoms(Derivative)) > 0:
        deriv = expr.atoms(Derivative).pop()
        expr = expr.subs(deriv, Function(str(deriv))(*deriv.free_symbols))
    while True:
        try:
            custom_fun = {
                _fun
                for _fun in expr.atoms(Function)
                if (_fun.class_key()[1] == 0)
                and (not _fun.class_key()[2] == "integral")
            }.pop()
            new_symbol_name = str(custom_fun)
            expr = expr.subs(custom_fun, Symbol(new_symbol_name))
        except KeyError:
            break
    return expr


class UnderlineDerivativePrinter(StrPrinter):
    def _print_Function(self, expr):
        return expr.func.__name__

    def _print_Derivative(self, expr):
        return "".join(
            [str(expr.args[0].func)]
            + [order * (DIFF_SYMBOL + str(key)) for key, order in expr.args[1:]]
        )


def sstr(expr, **settings):
    p = UnderlineDerivativePrinter(settings)
    s = p.doprint(expr)
    return s


Basic.__str__ = lambda self: sstr(self, order=None)

"""Convert sympy expression to np functions
todo: converges to torch_util

"""

import numpy as np
from sympy import lambdify
from typing import Iterable
from functools import reduce
import collections
from sympy import Max, Min, Mul

__all__ = ['lambdify_np']


class WrapSympy:
    is_sympy = True

    @staticmethod
    def _wrapper_guide(args):
        func_1 = args[0]
        func_2 = args[1]
        cond_1 = (isinstance(func_1, WrapSympy) and not func_1.is_sympy)
        cond_2 = isinstance(func_2, WrapSympy) and not func_2.is_sympy
        cond_3 = (not isinstance(func_1, WrapSympy)) and isinstance(func_1, collections.Callable)
        cond_4 = (not isinstance(func_2, WrapSympy)) and isinstance(func_2, collections.Callable)
        return cond_1 or cond_2 or cond_3 or cond_4, func_1, func_2


class WrapMax(Max, WrapSympy):
    def __new__(cls, *args, **kwargs):
        cond, func_1, func_2 = WrapMax._wrapper_guide(args)
        if cond:
            a = object.__new__(cls)
            a.f = func_1
            a.g = func_2
            a.is_sympy = False
        else:
            a = Max.__new__(cls, *args, **kwargs)
            if isinstance(a, WrapSympy):
                a.is_sympy = True
        return a

    def __call__(self, **x):
        if not self.is_sympy:
            f = lambdify_np(self.f, x.keys())
            g = lambdify_np(self.g, x.keys())
            return np.maximum(f(**x), g(**x))
        else:
            f = lambdify_np(self, x.keys())
            return f(**x)


class WrapMul(Mul, WrapSympy):
    def __new__(cls, *args, **kwargs):
        cond, func_1, func_2 = WrapMul._wrapper_guide(args)
        if cond:
            a = object.__new__(cls)
            a.f = func_1
            a.g = func_2
            a.is_sympy = False
        else:
            a = Mul.__new__(cls, *args, **kwargs)
            if isinstance(a, WrapSympy):
                a.is_sympy = True
        return a

    def __call__(self, **x):
        if not self.is_sympy:
            f = lambdify_np(self.f, x.keys())
            g = lambdify_np(self.g, x.keys())
            return f(**x) * g(**x)
        else:
            f = lambdify_np(self, x.keys())
            return f(**x)


class WrapMin(Min, WrapSympy):
    def __new__(cls, *args, **kwargs):
        cond, func_1, func_2 = WrapMin._wrapper_guide(args)
        if cond:
            a = object.__new__(cls)
            a.f = func_1
            a.g = func_2
            a.is_sympy = False
        else:
            a = Min.__new__(cls, *args, **kwargs)
            if isinstance(a, WrapSympy):
                a.is_sympy = True
        return a

    def __call__(self, **x):
        if not self.is_sympy:
            f = lambdify_np(self.f, x.keys())
            g = lambdify_np(self.g, x.keys())
            return np.minimum(f(**x), g(**x))
        else:
            f = lambdify_np(self, x.keys())
            return f(**x)


def _try_float(fn):
    try:
        fn = float(fn)
    except ValueError:
        pass
    except TypeError:
        pass
    return fn


def _constant_bool(boolean: bool):
    def fn(**x):
        return np.ones_like(next(iter(x.items()))[1], dtype=bool) if boolean else np.zeros_like(
            next(iter(x.items()))[1], dtype=bool)

    return fn


def _constant_float(f):
    def fn(**x):
        return np.ones_like(next(iter(x.items()))[1]) * f

    return fn


def lambdify_np(f, r: Iterable):
    if isinstance(r, dict):
        r = r.keys()
    if isinstance(f, WrapSympy) and f.is_sympy:
        lambdify_f = lambdify([k for k in r], f, [PLACEHOLDER, 'numpy'])
        lambdify_f.input_keys = [k for k in r]
        return lambdify_f
    if isinstance(f, WrapSympy) and not f.is_sympy:
        return f
    if isinstance(f, collections.Callable):
        return f
    if isinstance(f, bool):
        return _constant_bool(f)
    f = _try_float(f)
    if isinstance(f, float):
        return _constant_float(f)
    else:
        lambdify_f = lambdify([k for k in r], f, [PLACEHOLDER, 'numpy'])
        lambdify_f.input_keys = [k for k in r]
    return lambdify_f


PLACEHOLDER = {'amin': lambda x: reduce(lambda y, z: np.minimum(y, z), x),
               'amax': lambda x: reduce(lambda y, z: np.maximum(y, z), x),
               'Min': lambda *x: reduce(lambda y, z: np.minimum(y, z), x),
               'Max': lambda *x: reduce(lambda y, z: np.maximum(y, z), x),
               'Heaviside': lambda x: np.heaviside(x, 0),
               'equal': lambda x, y: np.isclose(x, y),
               'Xor': np.logical_xor,
               'cos': np.cos,
               'sin': np.sin,
               'tan': np.tan,
               'exp': np.exp,
               'sqrt': np.sqrt,
               'log': np.log,
               'sinh': np.sinh,
               'cosh': np.cosh,
               'tanh': np.tanh,
               'asin': np.arcsin,
               'acos': np.arccos,
               'atan': np.arctan,
               'Abs': np.abs,
               'DiracDelta': np.zeros_like,
               }

"""This module defines basic behaviour of Geometric Objects."""
import abc
import collections
import copy
import itertools
from functools import reduce
from typing import Dict, List, Union, Tuple
import numpy as np
from sympy import cos, sin, Symbol
import math

from idrlnet.geo_utils.sympy_np import lambdify_np, WrapMax, WrapMul, WrapMin


class CheckMeta(type):
    """Make sure that elements are checked when an instance is created,"""

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.check_elements()
        return obj


class AbsGeoObj(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def rotation(self, angle: float, axis: str = "z"):
        pass

    @abc.abstractmethod
    def scaling(self, scale: float):
        pass

    @abc.abstractmethod
    def translation(self, direction):
        pass


class Edge(AbsGeoObj):
    def __init__(self, functions, ranges: Dict, area):
        self.functions = functions
        self.ranges = ranges
        self.area = area

    @property
    def axes(self) -> List[str]:
        return [key for key in self.functions if not key.startswith("normal")]

    def rotation(self, angle: float, axis: str = "z"):
        assert len(self.axes) > 1, "Cannot rotate a object with dim<2"
        rotated_dims = [key for key in self.axes if key != axis]
        rd1, rd2, n = rotated_dims[0], rotated_dims[1], "normal_"
        self.functions[rd1] = (
            cos(angle) * self.functions[rd1] - sin(angle) * self.functions[rd2]
        )
        self.functions[n + rd1] = (
            cos(angle) * self.functions[n + rd1] - sin(angle) * self.functions[n + rd2]
        )
        self.functions[rd2] = (
            sin(angle) * self.functions[rd1] + cos(angle) * self.functions[rd2]
        )
        self.functions[n + rd2] = (
            sin(angle) * self.functions[n + rd1] + cos(angle) * self.functions[n + rd2]
        )
        return self

    def scaling(self, scale: float):
        for key in self.axes:
            self.functions[key] *= scale
        self.area = scale ** (len(self.axes) - 1) * self.area
        return self

    def translation(self, direction):
        assert len(direction) == len(
            self.axes
        ), "Moving direction must have the save dimension with the object"
        for key, x in zip(self.axes, direction):
            self.functions[key] += x
        return self

    def sample(
        self, density: int, param_ranges=None, low_discrepancy=False
    ) -> Dict[str, np.ndarray]:
        param_ranges = {} if param_ranges is None else param_ranges
        inputs = {**self.ranges, **param_ranges}.keys()
        area_fn = lambdify_np(self.area, inputs)
        param_points = _ranged_sample(100, ranges={**self.ranges, **param_ranges})
        nr_points = int(density * (np.mean(area_fn(**param_points))))

        lambdify_functions = {
            "area": lambda **x: area_fn(**x) / next(iter(x.values())).shape[0]
        }
        param_points = _ranged_sample(
            nr_points, {**self.ranges, **param_ranges}, low_discrepancy
        )
        data_var = {}

        for key, function in self.functions.items():
            lambdify_functions[key] = lambdify_np(function, inputs)

        for key, function in lambdify_functions.items():
            assert callable(function)
            data_var[key] = function(**param_points)

        for key in param_ranges:
            key = key if isinstance(key, str) else key.name
            data_var[key] = param_points[key]

        return data_var


class AbsCheckMix(abc.ABCMeta, CheckMeta):
    pass


class Geometry(AbsGeoObj, metaclass=AbsCheckMix):
    edges: List[Edge] = None
    bounds: Dict = None
    sdf = None

    def check_elements(self):
        if type(self) in [Geometry, Geometry1D, Geometry2D, Geometry3D]:
            return
        if self.edges is None:
            raise NotImplementedError("Geometry must define edges")
        if self.bounds is None:
            raise NotImplementedError("Geometry must define bounds")
        if self.sdf is None:
            raise NotImplementedError("Geometry must define sdf")

    @property
    def axes(self) -> List[str]:
        return self.edges[0].axes

    def translation(self, direction: Union[List, Tuple]) -> "Geometry":
        assert len(direction) == len(self.axes)
        [edge.translation(direction) for edge in self.edges]
        self.sdf = self.sdf.subs(
            [(Symbol(dim), Symbol(dim) - x) for dim, x in zip(self.axes, direction)]
        )
        self.bounds = {
            dim: (self.bounds[dim][0] + x, self.bounds[dim][1] + x)
            for dim, x in zip(self.axes, direction)
        }
        return self

    def rotation(self, angle: float, axis: str = "z", center=None) -> "Geometry":
        if center is not None:
            self.translation([-x for x in center])

        [edge.rotation(angle, axis) for edge in self.edges]
        rotated_dims = [key for key in self.axes if key != axis]
        sp_0 = Symbol(rotated_dims[0])
        _sp_0 = Symbol("tmp_0")
        sp_1 = Symbol(rotated_dims[1])
        _sp_1 = Symbol("tmp_1")
        self.sdf = self.sdf.subs(
            {
                sp_0: cos(angle) * _sp_0 + sin(angle) * _sp_1,
                sp_1: -sin(angle) * _sp_0 + cos(angle) * _sp_1,
            }
        )
        self.sdf = self.sdf.subs({_sp_0: sp_0, _sp_1: sp_1})
        self.bounds[rotated_dims[0]], self.bounds[rotated_dims[1]] = _rotate_rec(
            self.bounds[rotated_dims[0]], self.bounds[rotated_dims[1]], angle=angle
        )
        if center is not None:
            self.translation(center)
        return self

    def scaling(self, scale: float, center: Tuple = None) -> "Geometry":
        assert scale > 0, "scaling must be positive"
        if center is not None:
            self.translation(tuple([-x for x in center]))
        [edge.scaling(scale) for edge in self.edges]
        self.sdf = self.sdf.subs(
            {Symbol(dim): Symbol(dim) / scale for dim in self.axes}
        )
        self.sdf = scale * self.sdf
        for dim in self.axes:
            self.bounds[dim] = (
                self.bounds[dim][0] * scale,
                self.bounds[dim][1] * scale,
            )
        if center is not None:
            self.translation(center)
        return self

    def duplicate(self) -> "Geometry":
        return copy.deepcopy(self)

    def sample_boundary(
        self, density: int, sieve=None, param_ranges: Dict = None, low_discrepancy=False
    ) -> Dict[str, np.ndarray]:
        param_ranges = dict() if param_ranges is None else param_ranges
        points_list = [
            edge.sample(density, param_ranges, low_discrepancy) for edge in self.edges
        ]
        points = reduce(
            lambda e1, e2: {_k: np.concatenate([e1[_k], e2[_k]], axis=0) for _k in e1},
            points_list,
        )
        points = self._sieve_points(points, sieve, sign=-1, tol=1e-4)
        return points

    def _sieve_points(self, points, sieve, tol=1e-4, sign=1.0):

        sdf_fn = lambdify_np(self.sdf, points.keys())
        points["sdf"] = sdf_fn(**points)

        criteria_fn = lambdify_np(True if sieve is None else sieve, points.keys())
        criteria_index = np.logical_and(
            np.greater(points["sdf"], -tol), criteria_fn(**points)
        )
        if sign == -1:
            criteria_index = np.logical_and(np.less(points["sdf"], tol), criteria_index)
        points = {k: v[criteria_index[:, 0], :] for k, v in points.items()}
        return points

    def sample_interior(
        self,
        density: int,
        bounds: Dict = None,
        sieve=None,
        param_ranges: Dict = None,
        low_discrepancy=False,
    ) -> Dict[str, np.ndarray]:
        bounds = self.bounds if bounds is None else bounds
        bounds = {
            Symbol(key) if isinstance(key, str) else key: value
            for key, value in bounds.items()
        }
        param_ranges = {} if param_ranges is None else param_ranges
        measure = np.prod([value[1] - value[0] for value in bounds.values()])
        nr_points = int(measure * density)

        points = _ranged_sample(
            nr_points, {**bounds, **param_ranges}, low_discrepancy=low_discrepancy
        )
        assert len(points.keys()) >= 0, "No points have been sampled!"

        points = self._sieve_points(points, sieve, tol=0.0)

        points["area"] = np.zeros_like(points["sdf"]) + (1.0 / density)
        return points

    def __add__(self, other: "Geometry") -> "Geometry":
        geo = self.generate_geo_obj(other)
        geo.edges = self.edges + other.edges
        geo.sdf = WrapMax(self.sdf, other.sdf)
        geo.bounds = dict()
        for key, value in self.bounds.items():
            geo.bounds[key] = (
                min(other.bounds[key][0], self.bounds[key][0]),
                max(other.bounds[key][1], self.bounds[key][1]),
            )
        return geo

    def generate_geo_obj(self, other=None):
        if isinstance(self, Geometry1D):
            geo = Geometry1D()
            assert isinstance(other, Geometry1D) or other is None
        elif isinstance(self, Geometry2D):
            geo = Geometry2D()
            assert isinstance(other, Geometry2D) or other is None
        elif isinstance(self, Geometry3D):
            geo = Geometry3D()
            assert isinstance(other, Geometry3D) or other is None
        else:
            raise TypeError
        return geo

    def __sub__(self, other: "Geometry") -> "Geometry":
        geo = self.generate_geo_obj(other)

        geo.edges = self.edges + [_inverse_edge(edge) for edge in other.edges]
        geo.sdf = WrapMin(self.sdf, WrapMul(-1, other.sdf))
        geo.bounds = dict()
        for key, value in self.bounds.items():
            geo.bounds[key] = (self.bounds[key][0], self.bounds[key][1])
        return geo

    def __invert__(self) -> "Geometry":
        geo = self.generate_geo_obj()
        geo.edges = [_inverse_edge(edge) for edge in self.edges]
        geo.sdf = WrapMul(-1, self.sdf)
        for key, value in self.bounds.items():
            geo.bounds[key] = (-float("inf"), float("inf"))
        return geo

    def __and__(self, other: "Geometry") -> "Geometry":
        geo = self.generate_geo_obj(other)
        geo.edges = self.edges + other.edges
        geo.sdf = WrapMin(self.sdf, other.sdf)
        geo.bounds = dict()
        for key, value in self.bounds.items():
            geo.bounds[key] = (
                max(other.bounds[key][0], self.bounds[key][0]),
                min(other.bounds[key][1], self.bounds[key][1]),
            )
        return geo


class Geometry1D(Geometry):
    pass


class Geometry2D(Geometry):
    pass


class Geometry3D(Geometry):
    pass


# todo: sample in cuda device
def _ranged_sample(
    batch_size: int, ranges: Dict, low_discrepancy: bool = False
) -> Dict[str, np.ndarray]:
    points = dict()
    low_discrepancy_stack = []
    for key, value in ranges.items():
        if isinstance(value, (float, int)):
            samples = np.ones((batch_size, 1)) * value
        elif isinstance(value, tuple):
            assert len(value) == 2, "Tuple: length of range should be 2!"
            if low_discrepancy:
                low_discrepancy_stack.append((key.name, value))
                continue
            else:
                samples = np.random.uniform(value[0], value[1], size=(batch_size, 1))
        elif isinstance(value, collections.Callable):
            samples = value(batch_size)
        else:
            raise TypeError(f"range type {type(value)} not supported!")
        points[key.name] = samples
    if low_discrepancy:
        low_discrepancy_points_dict = _low_discrepancy_sampling(
            batch_size, low_discrepancy_stack
        )
        points.update(low_discrepancy_points_dict)
    for key, v in points.items():
        points[key] = v.astype(np.float64)
    return points


def _rotate_rec(x: Tuple, y: Tuple, angle: float):
    points = itertools.product(x, y)
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = -float("inf"), -float("inf")
    try:
        for x, y in points:
            new_x = cos(angle) * x - sin(angle) * y
            new_y = sin(angle) * x + cos(angle) * y
            min_x = min(new_x, min_x)
            min_y = min(new_y, min_y)
            max_x = max(new_x, max_x)
            max_y = max(new_y, max_y)
    except TypeError:
        angle = math.pi / 4
        for x, y in points:
            new_x = cos(angle) * x - sin(angle) * y
            new_y = sin(angle) * x + cos(angle) * y
            min_x = min(new_x, min_x)
            min_y = min(new_y, min_y)
            max_x = max(new_x, max_x)
            max_y = max(new_y, max_y)
    return (min_x, max_x), (min_y, max_y)


def _low_discrepancy_sampling(n_points, low_discrepancy_stack: List[Tuple]):
    dim = len(low_discrepancy_stack)
    sections = 2 ** dim

    def uniform(x, start, end, rmin, bi_range=0.5):
        dims = len(rmin)
        if end - start <= 1:
            return
        d, r = (end - start) // sections, (end - start) % sections
        r = (np.arange(sections - 1, 0, -1) + r) // sections
        np.random.shuffle(r)
        d = (d + r).cumsum() + start
        q = np.concatenate([np.array([start]), d, np.array([end])])

        for i in range(len(q) - 1):
            for j in range(dims):
                x[q[i] : q[i + 1], j] = (
                    (x[q[i] : q[i + 1], j] - rmin[j]) / 2
                    + rmin[j]
                    + ((i >> j) & 1) * bi_range
                )
            rmin_sub = [v + bi_range * ((i >> j) & 1) for j, v in enumerate(rmin)]
            uniform(x, q[i], q[i + 1], rmin_sub, bi_range=bi_range / 2)
        return x

    n = n_points
    points = np.random.rand(n, dim)
    uniform(points, start=0, end=n, rmin=[0] * dim)
    points_dict = {}
    for i, (key, bi_range) in enumerate(low_discrepancy_stack):
        points_dict[key] = (
            points[:, i : i + 1] * (bi_range[1] - bi_range[0]) + bi_range[0]
        )
    return points_dict


def _inverse_edge(edge: Edge):
    new_functions = {
        k: -v if k.startswith("normal_") else v for k, v in edge.functions.items()
    }
    edge = Edge(functions=new_functions, ranges=edge.ranges, area=edge.area)
    return edge

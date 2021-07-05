"""Concrete shape."""

import math
from math import pi
from typing import Union, List, Tuple
import numpy as np
from sympy import symbols, Abs, sqrt, Max, Min, cos, sin, log, sign, Heaviside
from sympy.vector import CoordSys3D
from .geo import Edge, Geometry1D, Geometry2D, Geometry3D

__all__ = ['Line1D', 'Line', 'Tube2D', 'Rectangle', 'Circle', 'Heart', 'Triangle', 'Polygon', 'Plane', 'Tube3D', 'Tube',
           'CircularTube', 'Box', 'Sphere', 'Cylinder']


class Line1D(Geometry1D):

    def __init__(self, point_1, point_2):
        x, none = symbols('x none')
        ranges = {none: (0, 1)}
        edge_1 = Edge(functions={'x': point_1,
                                 'normal_x': -1},
                      area=1.0,
                      ranges=ranges)
        edge_2 = Edge(functions={'x': point_2,
                                 'normal_x': 1},
                      area=1.0,
                      ranges=ranges)
        self.edges = [edge_1, edge_2]
        dist = point_2 - point_1
        center_x = point_1 + dist / 2
        self.sdf = dist / 2 - Abs(x - center_x)

        self.bounds = {'x': (point_1, point_2)}


class Line(Geometry2D):
    def __init__(self, point_1, point_2, normal=1):
        x, y, l = symbols('x y l')
        ranges = {l: (0, 1)}
        dist_x = point_2[0] - point_1[0]
        dist_y = point_2[1] - point_1[1]
        normal_vector = (-dist_y * normal, dist_x * normal)
        normal_norm = math.sqrt(normal_vector[0] ** 2 + normal_vector[1] ** 2)
        normal_vector = (normal_vector[0] / normal_norm, normal_vector[1] / normal_norm)
        line_1 = Edge(functions={'x': point_1[0] + l * dist_x,
                                 'y': point_1[1] + l * dist_y,
                                 'normal_x': normal_vector[0],
                                 'normal_y': normal_vector[1]},
                      ranges=ranges,
                      area=normal_norm)
        self.edges = [line_1]
        self.sdf = ((x - point_1[0]) * dist_y - (y - point_1[1]) * dist_x) / normal_norm
        self.bounds = {'x': (min(point_1[0], point_2[0]), max(point_1[0], point_2[0])),
                       'y': (min(point_1[1], point_2[1]), max(point_1[1], point_2[1]))}


class Tube2D(Geometry2D):

    def __init__(self, point_1, point_2):
        l, y = symbols('l y')
        ranges = {l: (0, 1)}
        dist_x = point_2[0] - point_1[0]
        dist_y = point_2[1] - point_1[1]
        line_1 = Edge(functions={'x': l * dist_x + point_1[0],
                                 'y': point_1[1],
                                 'normal_x': 0,
                                 'normal_y': -1},
                      ranges=ranges,
                      area=dist_x)
        line_2 = Edge(functions={'x': l * dist_x + point_1[0],
                                 'y': point_2[1],
                                 'normal_x': 0,
                                 'normal_y': 1},
                      ranges=ranges,
                      area=dist_x)
        self.edges = [line_1, line_2]
        center_y = point_1[1] + (dist_y) / 2
        y_diff = Abs(y - center_y) - (point_2[1] - center_y)
        outside_distance = sqrt(Max(y_diff, 0) ** 2)
        inside_distance = Min(y_diff, 0)
        self.sdf = - (outside_distance + inside_distance)
        self.bounds = {'x': (min(point_1[0], point_2[0]), max(point_1[0], point_2[0])),
                       'y': (min(point_1[1], point_2[1]), max(point_1[1], point_2[1]))}


class Rectangle(Geometry2D):
    def __init__(self, point_1, point_2):
        l, x, y = symbols('l x y')
        ranges = {l: (0, 1)}
        dist_x = point_2[0] - point_1[0]
        dist_y = point_2[1] - point_1[1]

        edge_1 = Edge(functions={'x': l * dist_x + point_1[0],
                                 'y': point_1[1],
                                 'normal_x': 0,
                                 'normal_y': -1},
                      ranges=ranges,
                      area=dist_x)
        edge_2 = Edge(functions={'x': point_2[0],
                                 'y': l * dist_y + point_1[1],
                                 'normal_x': 1,
                                 'normal_y': 0},
                      ranges=ranges,
                      area=dist_y)
        edge_3 = Edge(functions={'x': l * dist_x + point_1[0],
                                 'y': point_2[1],
                                 'normal_x': 0,
                                 'normal_y': 1},
                      ranges=ranges,
                      area=dist_x)
        edge_4 = Edge(functions={'x': point_1[0],
                                 'y': -l * dist_y + point_2[1],
                                 'normal_x': -1,
                                 'normal_y': 0},
                      ranges=ranges,
                      area=dist_y)
        self.edges = [edge_1, edge_2, edge_3, edge_4]
        center_x = point_1[0] + (dist_x) / 2
        center_y = point_1[1] + (dist_y) / 2
        x_diff = Abs(x - center_x) - (point_2[0] - center_x)
        y_diff = Abs(y - center_y) - (point_2[1] - center_y)
        outside_distance = sqrt(Max(x_diff, 0) ** 2 + Max(y_diff, 0) ** 2)
        inside_distance = Min(Max(x_diff, y_diff), 0)
        self.sdf = - (outside_distance + inside_distance)
        self.bounds = {'x': (min(point_1[0], point_2[0]), max(point_1[0], point_2[0])),
                       'y': (min(point_1[1], point_2[1]), max(point_1[1], point_2[1]))}


class Circle(Geometry2D):

    def __init__(self, center, radius):
        theta, x, y = symbols('theta x y')
        ranges = {theta: (0, 2 * math.pi)}
        edge = Edge(functions={'x': center[0] + radius * cos(theta),
                               'y': center[1] + radius * sin(theta),
                               'normal_x': 1 * cos(theta),
                               'normal_y': 1 * sin(theta)},
                    ranges=ranges,
                    area=2 * pi * radius)

        self.edges = [edge]
        self.sdf = radius - sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        self.bounds = {'x': (center[0] - radius, center[0] + radius), 'y': (center[1] - radius, center[1] + radius)}


class Heart(Geometry2D):
    def __init__(self, center=(0, 0.5), radius=0.5):
        c1, c2 = center
        theta, t, x, y = symbols('t theta x y')
        ranges = {theta: (0, math.pi), t: (0, 1)}
        edge_1 = Edge(functions={'x': center[0] - t * radius,
                                 'y': center[1] - (1 - t) * radius,
                                 'normal_x': -1.,
                                 'normal_y': -1.},
                      ranges=ranges,
                      area=math.sqrt(2) * radius)

        edge_2 = Edge(functions={'x': center[0] + t * radius,
                                 'y': center[1] - (1 - t) * radius,
                                 'normal_x': 1.,
                                 'normal_y': -1.},
                      ranges=ranges,
                      area=math.sqrt(2) * radius)

        edge_3 = Edge(functions={'x': center[0] - radius / 2 + radius / math.sqrt(2) * cos(math.pi / 4 * 5 - theta),
                                 'y': center[1] + radius / 2 + radius / math.sqrt(2) * sin(math.pi / 4 * 5 - theta),
                                 'normal_x': cos(math.pi / 4 * 5 - theta),
                                 'normal_y': sin(math.pi / 4 * 5 - theta)},
                      ranges=ranges,
                      area=math.sqrt(2) * radius * math.pi)

        edge_4 = Edge(functions={'x': center[0] + radius / 2 + radius / math.sqrt(2) * cos(math.pi / 4 * 3 - theta),
                                 'y': center[1] + radius / 2 + radius / math.sqrt(2) * sin(math.pi / 4 * 3 - theta),
                                 'normal_x': cos(math.pi / 4 * 3 - theta),
                                 'normal_y': sin(math.pi / 4 * 3 - theta)},
                      ranges=ranges,
                      area=math.sqrt(2) * radius * math.pi)

        self.edges = [edge_1, edge_2, edge_3, edge_4]
        x, y = symbols('x y')
        x = (x - c1) * 0.5 / radius
        y = (y - c2) * 0.5 / radius + 0.5
        part1 = Heaviside(Abs(x) + y - 1) * (sqrt((Abs(x) - 0.25) ** 2 + (y - 0.75) ** 2) - math.sqrt(2) / 4)
        part_i = 0.5 * Max(Abs(x) + y, 0)
        part2 = (1 - Heaviside(Abs(x) + y - 1)) * sign(Abs(x) - y) * Min(sqrt(Abs(x) ** 2 + (y - 1) ** 2),
                                                                         sqrt((Abs(x) - part_i) ** 2 + (
                                                                                 y - part_i) ** 2))
        self.sdf = (-part1 - part2) * radius * 2
        self.bounds = {'x': (
            center[0] - 0.5 * radius - 0.5 * math.sqrt(2) * radius,
            center[0] + 0.5 * radius + 0.5 * math.sqrt(2) * radius),
            'y': (center[1] - radius, center[1] + 0.5 * radius + 0.5 * math.sqrt(2) * radius)}


class Triangle(Geometry2D):
    def __init__(self, p0, p1, p2):
        x, y, t = symbols('x y t')
        N = CoordSys3D('N')
        P0 = p0[0] * N.i + p0[1] * N.j
        P1 = p1[0] * N.i + p1[1] * N.j
        P2 = p2[0] * N.i + p2[1] * N.j
        p = x * N.i + y * N.j
        e0, e1, e2 = P1 - P0, P2 - P1, P0 - P2
        v0, v1, v2 = p - P0, p - P1, p - P2
        pq0 = v0 - e0 * Max(Min(v0.dot(e0) / e0.dot(e0), 1), 0)
        pq1 = v1 - e1 * Max(Min(v1.dot(e1) / e1.dot(e1), 1), 0)
        pq2 = v2 - e2 * Max(Min(v2.dot(e2) / e2.dot(e2), 1), 0)
        s = sign(e0.dot(N.i) * e2.dot(N.j) - e0.dot(N.j) * e2.dot(N.i))

        u = sqrt(Min(pq0.dot(pq0), pq1.dot(pq1), pq2.dot(pq2)))

        v = Min(s * (v0.dot(N.i) * e0.dot(N.j) - v0.dot(N.j) * e0.dot(N.i)),
                s * (v1.dot(N.i) * e1.dot(N.j) - v1.dot(N.j) * e1.dot(N.i)),
                s * (v2.dot(N.i) * e2.dot(N.j) - v2.dot(N.j) * e2.dot(N.i)))
        self.sdf = u * sign(v)

        l0 = sqrt(e0.dot(e0))
        l1 = sqrt(e1.dot(e1))
        l2 = sqrt(e2.dot(e2))
        ranges = {t: (0, 1)}
        in_out_sign = -sign(e0.cross(e1).dot(N.k))
        edge_1 = Edge(functions={'x': p1[0] + t * (p0[0] - p1[0]),
                                 'y': p1[1] + t * (p0[1] - p1[1]),
                                 'normal_x': (p0[1] - p1[1]) / l0 * in_out_sign,
                                 'normal_y': (p1[0] - p0[0]) / l0 * in_out_sign},
                      ranges=ranges,
                      area=l0)
        edge_2 = Edge(functions={'x': p2[0] + t * (p1[0] - p2[0]),
                                 'y': p2[1] + t * (p1[1] - p2[1]),
                                 'normal_x': (p1[1] - p2[1]) / l1 * in_out_sign,
                                 'normal_y': (p2[0] - p1[0]) / l1 * in_out_sign},
                      ranges=ranges,
                      area=l1)
        edge_3 = Edge(functions={'x': p0[0] + t * (p2[0] - p0[0]),
                                 'y': p0[1] + t * (p2[1] - p0[1]),
                                 'normal_x': (p2[1] - p0[1]) / l2 * in_out_sign,
                                 'normal_y': (p0[0] - p2[0]) / l2 * in_out_sign},
                      ranges=ranges,
                      area=l2)
        self.edges = [edge_1, edge_2, edge_3]
        self.bounds = {'x': (min(p0[0], p1[0], p2[0]), max(p0[0], p1[0], p2[0])),
                       'y': (min(p0[1], p1[1], p2[1]), max(p0[1], p1[1], p2[1]))}


class Polygon(Geometry2D):
    def __init__(self, points):
        v = points
        t = symbols('t')
        ranges = {t: (0, 1)}

        def _sdf(x: np.ndarray, y: np.ndarray, **kwargs):
            s = np.ones_like(x)
            _points = np.concatenate([x, y], axis=1)
            d = ((np.array(v[0]) - _points) ** 2).sum(axis=1, keepdims=True)
            for i in range(len(v)):
                e = np.array(v[i - 1]) - np.array(v[i])
                w = _points - np.array(v[i])
                b = w - e * np.clip((w * e).sum(axis=1, keepdims=True) / (e * e).sum(), 0, 1)
                d = np.minimum(d, (b * b).sum(keepdims=True, axis=1))
                cond1 = _points[:, 1:] >= v[i][1]
                cond2 = _points[:, 1:] < v[i - 1][1]
                cond3 = e[0] * w[:, 1:] > e[1] * w[:, :1]
                inverse_idx1 = np.all([cond1, cond2, cond3], axis=0)
                inverse_idx2 = np.all([np.logical_not(cond1), np.logical_not(cond2), np.logical_not(cond3)], axis=0)
                inverse_idx = np.any([inverse_idx1, inverse_idx2], axis=0)
                s[inverse_idx] *= -1
            return -np.sqrt(d) * s

        self.sdf = _sdf
        self.edges = []
        for i, _ in enumerate(points):
            length = math.sqrt((points[i - 1][0] - points[i][0]) ** 2 + (points[i - 1][1] - points[i][1]) ** 2)
            edge = Edge(functions={'x': points[i - 1][0] - t * (points[i - 1][0] - points[i][0]),
                                   'y': points[i - 1][1] - t * (points[i - 1][1] - points[i][1]),
                                   'normal_x': (points[i][1] - points[i - 1][1]) / length,
                                   'normal_y': (points[i - 1][0] - points[i][0]) / length},
                        ranges=ranges,
                        area=length)
            self.edges.append(edge)
        _p = iter(zip(*points))
        _p1 = next(_p)
        _p2 = next(_p)
        self.bounds = {'x': (min(_p1), max(_p1)),
                       'y': (min(_p2), max(_p2))}

    def translation(self, direction: Union[List, Tuple]):
        raise NotImplementedError

    def rotation(self, angle: float, axis: str = 'z', center=None):
        raise NotImplementedError

    def scaling(self, scale: float, center: Tuple = None):
        raise NotImplementedError


class Plane(Geometry3D):

    def __init__(self, point_1, point_2, normal):
        assert point_1[0] == point_2[0], "Points must have the same x coordinate"

        x, y, z, s_1, s_2 = symbols('x y z s_1 s_2')
        center = (point_1[0] + (point_2[0] - point_1[0]) / 2,
                  point_1[1] + (point_2[1] - point_1[1]) / 2,
                  point_1[2] + (point_2[2] - point_1[2]) / 2)
        side_y = point_2[1] - point_1[1]
        side_z = point_2[2] - point_1[2]

        ranges = {s_1: (-1, 1), s_2: (-1, 1)}
        edge = Edge(functions={'x': center[0],
                               'y': center[1] + 0.5 * s_1 * side_y,
                               'z': center[2] + 0.5 * s_2 * side_z,
                               'normal_x': 1e-10 + normal,  # TODO rm 1e-10
                               'normal_y': 0,
                               'normal_z': 0},
                    ranges=ranges,
                    area=side_y * side_z)
        self.edges = [edge]

        self.sdf = normal * (center[0] - x)

        self.bounds = {'x': (min(point_1[0], point_2[0]), max(point_1[0], point_2[0])),
                       'y': (min(point_1[1], point_2[1]), max(point_1[1], point_2[1])),
                       'z': (min(point_1[2], point_2[2]), max(point_1[2], point_2[2])), }


class Tube3D(Geometry3D):

    def __init__(self, point_1, point_2):
        x, y, z, s_1, s_2 = symbols('x y z s_1 s_2')
        center = (point_1[0] + (point_2[0] - point_1[0]) / 2,
                  point_1[1] + (point_2[1] - point_1[1]) / 2,
                  point_1[2] + (point_2[2] - point_1[2]) / 2)
        side_x = point_2[0] - point_1[0]
        side_y = point_2[1] - point_1[1]
        side_z = point_2[2] - point_1[2]

        ranges = {s_1: (-1, 1), s_2: (-1, 1)}
        edge_1 = Edge(functions={'x': center[0] + 0.5 * s_1 * side_x,
                                 'y': center[1] + 0.5 * s_2 * side_y,
                                 'z': center[2] + 0.5 * side_z,
                                 'normal_x': 0,
                                 'normal_y': 0,
                                 'normal_z': 1},
                      ranges=ranges,
                      area=side_x * side_y)
        edge_2 = Edge(functions={'x': center[0] + 0.5 * s_1 * side_x,
                                 'y': center[1] + 0.5 * s_2 * side_y,
                                 'z': center[2] - 0.5 * side_z,
                                 'normal_x': 0,
                                 'normal_y': 0,
                                 'normal_z': -1},
                      ranges=ranges,
                      area=side_x * side_y)
        edge_3 = Edge(functions={'x': center[0] + 0.5 * s_1 * side_x,
                                 'y': center[1] + 0.5 * side_y,
                                 'z': center[2] + 0.5 * s_2 * side_z,
                                 'normal_x': 0,
                                 'normal_y': 1,
                                 'normal_z': 0},
                      ranges=ranges,
                      area=side_x * side_z)
        edge_4 = Edge(functions={'x': center[0] + 0.5 * s_1 * side_x,
                                 'y': center[1] - 0.5 * side_y,
                                 'z': center[2] + 0.5 * s_2 * side_z,
                                 'normal_x': 0,
                                 'normal_y': -1,
                                 'normal_z': 0},
                      ranges=ranges,
                      area=side_x * side_z)
        self.edges = [edge_1, edge_2, edge_3, edge_4]
        y_dist = Abs(y - center[1]) - 0.5 * side_y
        z_dist = Abs(z - center[2]) - 0.5 * side_z
        outside_distance = sqrt(Max(y_dist, 0) ** 2 + Max(z_dist, 0) ** 2)
        inside_distance = Min(Max(y_dist, z_dist), 0)
        self.sdf = - (outside_distance + inside_distance)

        self.bounds = {'x': (min(point_1[0], point_2[0]), max(point_1[0], point_2[0])),
                       'y': (min(point_1[1], point_2[1]), max(point_1[1], point_2[1])),
                       'z': (min(point_1[2], point_2[2]), max(point_1[2], point_2[2])), }


class Tube(Tube3D):
    def __init__(self, point_1, point_2):
        super(Tube, self).__init__(point_1, point_2)


class CircularTube(Geometry3D):
    def __init__(self, center, radius, height):
        x, y, z, h, theta = symbols('x y z h theta')
        ranges = {h: (-1, 1), theta: (0, 2 * pi)}
        edge_1 = Edge(functions={'x': center[0] + radius * cos(theta),
                                 'y': center[1] + radius * sin(theta),
                                 'z': center[2] + 0.5 * h * height,
                                 'normal_x': 1 * cos(theta),
                                 'normal_y': 1 * sin(theta),
                                 'normal_z': 0},
                      ranges=ranges,
                      area=height * 2 * pi * radius)

        self.edges = [edge_1]
        self.sdf = radius - sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        self.bounds = {'x': (center[0] - radius, center[0] + radius),
                       'y': (center[1] - radius, center[1] + radius),
                       'z': (center[2] - height / 2, center[2] + height / 2)}


class Box(Geometry3D):
    def __init__(self, point_1, point_2):
        x, y, z, s_1, s_2 = symbols('x y z s_1 s_2')
        center = (point_1[0] + (point_2[0] - point_1[0]) / 2,
                  point_1[1] + (point_2[1] - point_1[1]) / 2,
                  point_1[2] + (point_2[2] - point_1[2]) / 2)
        side_x = point_2[0] - point_1[0]
        side_y = point_2[1] - point_1[1]
        side_z = point_2[2] - point_1[2]

        ranges = {s_1: (-1, 1), s_2: (-1, 1)}

        self.bounds = {'x': (min(point_1[0], point_2[0]), max(point_1[0], point_2[0])),
                       'y': (min(point_1[1], point_2[1]), max(point_1[1], point_2[1])),
                       'z': (min(point_1[2], point_2[2]), max(point_1[2], point_2[2])), }

        edge_1 = Edge(functions={'x': center[0] + 0.5 * s_1 * side_x,
                                 'y': center[1] + 0.5 * s_2 * side_y,
                                 'z': center[2] + 0.5 * side_z,
                                 'normal_x': 0,
                                 'normal_y': 0,
                                 'normal_z': 1},
                      ranges=ranges,
                      area=side_x * side_y)
        edge_2 = Edge(functions={'x': center[0] + 0.5 * s_1 * side_x,
                                 'y': center[1] + 0.5 * s_2 * side_y,
                                 'z': center[2] - 0.5 * side_z,
                                 'normal_x': 0,
                                 'normal_y': 0,
                                 'normal_z': -1},
                      ranges=ranges,
                      area=side_x * side_y)
        edge_3 = Edge(functions={'x': center[0] + 0.5 * s_1 * side_x,
                                 'y': center[1] + 0.5 * side_y,
                                 'z': center[2] + 0.5 * s_2 * side_z,
                                 'normal_x': 0,
                                 'normal_y': 1,
                                 'normal_z': 0},
                      ranges=ranges,
                      area=side_x * side_z)
        edge_4 = Edge(functions={'x': center[0] + 0.5 * s_1 * side_x,
                                 'y': center[1] - 0.5 * side_y,
                                 'z': center[2] + 0.5 * s_2 * side_z,
                                 'normal_x': 0,
                                 'normal_y': -1,
                                 'normal_z': 0},
                      ranges=ranges,
                      area=side_x * side_z)
        edge_5 = Edge(functions={'x': center[0] + 0.5 * side_x,
                                 'y': center[1] + 0.5 * s_1 * side_y,
                                 'z': center[2] + 0.5 * s_2 * side_z,
                                 'normal_x': 1,
                                 'normal_y': 0,
                                 'normal_z': 0},
                      ranges=ranges,
                      area=side_y * side_z)
        edge_6 = Edge(functions={'x': center[0] - 0.5 * side_x,
                                 'y': center[1] + 0.5 * s_1 * side_y,
                                 'z': center[2] + 0.5 * s_2 * side_z,
                                 'normal_x': -1,
                                 'normal_y': 0,
                                 'normal_z': 0},
                      ranges=ranges,
                      area=side_y * side_z)
        self.edges = [edge_1, edge_2, edge_3, edge_4, edge_5, edge_6]
        x_dist = Abs(x - center[0]) - 0.5 * side_x
        y_dist = Abs(y - center[1]) - 0.5 * side_y
        z_dist = Abs(z - center[2]) - 0.5 * side_z
        outside_distance = sqrt(Max(x_dist, 0) ** 2 + Max(y_dist, 0) ** 2 + Max(z_dist, 0) ** 2)
        inside_distance = Min(Max(x_dist, y_dist, z_dist), 0)
        self.sdf = - (outside_distance + inside_distance)
        self.bounds = {'x': (min(point_1[0], point_2[0]), max(point_1[0], point_2[0])),
                       'y': (min(point_1[1], point_2[1]), max(point_1[1], point_2[1])),
                       'z': (min(point_1[2], point_2[2]), max(point_1[2], point_2[2])), }


class Sphere(Geometry3D):

    def __init__(self, center, radius):
        x, y, z, v_1, v_2, u_1, u_2 = symbols('x y z v_1 v_2 u_1 u_2')
        ranges = {v_1: (0, 1), v_2: (0, 1), u_1: (0, 1), u_2: (0, 1)}
        r_1 = sqrt(-log(v_1)) * cos(2 * pi * u_1)
        r_2 = sqrt(-log(v_1)) * sin(2 * pi * u_1)
        r_3 = sqrt(-log(v_2)) * cos(2 * pi * u_2)

        norm = sqrt(r_1 ** 2 + r_2 ** 2 + r_3 ** 2)
        edge_1 = Edge(functions={'x': center[0] + radius * r_1 / norm,
                                 'y': center[1] + radius * r_2 / norm,
                                 'z': center[2] + radius * r_3 / norm,
                                 'normal_x': r_1 / norm,
                                 'normal_y': r_2 / norm,
                                 'normal_z': r_3 / norm},
                      ranges=ranges,
                      area=4 * pi * radius ** 2)

        self.edges = [edge_1]
        self.sdf = radius - sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
        self.bounds = {'x': (center[0] - radius, center[0] + radius),
                       'y': (center[1] - radius, center[1] + radius),
                       'z': (center[2] - radius, center[2] + radius)}


class Cylinder(Geometry3D):

    def __init__(self, center, radius, height):
        x, y, z, h, r, theta = symbols('x y z h r theta')
        ranges = {h: (-1, 1), r: (0, 1), theta: (0, 2 * pi)}
        edge_1 = Edge(functions={'x': center[0] + radius * cos(theta),
                                 'y': center[1] + radius * sin(theta),
                                 'z': center[2] + 0.5 * h * height,
                                 'normal_x': 1 * cos(theta),
                                 'normal_y': 1 * sin(theta),
                                 'normal_z': 0},
                      ranges=ranges,
                      area=height * 2 * pi * radius)
        edge_2 = Edge(functions={'x': center[0] + sqrt(r) * radius * cos(theta),
                                 'y': center[1] + sqrt(r) * radius * sin(theta),
                                 'z': center[2] + 0.5 * height,
                                 'normal_x': 0,
                                 'normal_y': 0,
                                 'normal_z': 1},
                      ranges=ranges,
                      area=math.pi * radius ** 2)
        edge_3 = Edge(functions={'x': center[0] + sqrt(r) * radius * cos(theta),
                                 'y': center[1] + sqrt(r) * radius * sin(theta),
                                 'z': center[2] - 0.5 * height,
                                 'normal_x': 0,
                                 'normal_y': 0,
                                 'normal_z': -1},
                      ranges=ranges,
                      area=pi * radius ** 2)
        self.edges = [edge_1, edge_2, edge_3]

        r_dist = sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        z_dist = Abs(z - center[2])
        outside_distance = sqrt(Min(0, radius - r_dist) ** 2 + Min(0, 0.5 * height - z_dist) ** 2)
        inside_distance = -1 * Min(Abs(Min(0, r_dist - radius)), Abs(Min(0, z_dist - 0.5 * height)))
        self.sdf = - (outside_distance + inside_distance)

        self.bounds = {'x': (center[0] - radius, center[0] + radius),
                       'y': (center[1] - radius, center[1] + radius),
                       'z': (center[2] - height / 2, center[2] + height / 2)}

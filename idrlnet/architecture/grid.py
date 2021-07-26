""" The module is experimental. It may be removed or totally refactored in the future."""

import idrlnet.architecture.mlp as mlp
import itertools
import torch
from typing import List, Tuple, Union, Dict
from idrlnet.geo_utils.geo_obj import Rectangle
from idrlnet.net import NetNode
from idrlnet.pde_op.operator import Difference
from idrlnet.data import get_data_node


def indicator(xn: torch.Tensor, *axis_bounds):
    # todo: use `heavyside`
    i = 0
    lb, ub, lb_eq = axis_bounds[0]
    if lb_eq:
        indic = torch.logical_and(
            xn[:, i : i + 1] >= axis_bounds[0][0], axis_bounds[0][1] >= xn[:, i : i + 1]
        )
    else:
        indic = torch.logical_and(
            xn[:, i : i + 1] > axis_bounds[0][0], axis_bounds[0][1] >= xn[:, i : i + 1]
        )
    for i, (lb, ub, lb_eq) in enumerate(axis_bounds[1:]):
        if lb_eq:
            indic = torch.logical_and(
                indic,
                torch.logical_and(
                    xn[:, i + 1 : i + 2] >= lb, ub >= xn[:, i + 1 : i + 2]
                ),
            )
        else:
            indic = torch.logical_and(
                indic,
                torch.logical_and(
                    xn[:, i + 1 : i + 2] > lb, ub >= xn[:, i + 1 : i + 2]
                ),
            )
    return indic


class NetEval(torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, columns, rows, **kwargs):
        super().__init__()
        self.columns = columns
        self.rows = rows
        self.n_columns = len(self.columns) - 1
        self.n_rows = len(self.rows) - 1
        self.nets = []
        if "net_generator" in kwargs.keys():
            net_gen = kwargs.pop("net_generator")
        else:
            net_gen = lambda: mlp.MLP([n_inputs, 20, 20, 20, 20, n_outputs])
        for i in range(self.n_columns):
            self.nets.append([])
        for i in range(self.n_columns):
            for j in range(self.n_rows):
                self.nets[i].append(net_gen())
        self.layers = torch.nn.ModuleList(itertools.chain(*self.nets))

    def forward(self, x):
        xn = x.detach()
        y = 0
        for i in range(self.n_columns):
            for j in range(self.n_rows):
                y += (
                    indicator(
                        xn,
                        (
                            self.columns[i],
                            self.columns[i + 1],
                            True if i == 0 else False,
                        ),
                        (self.rows[j], self.rows[j + 1], True if j == 0 else False),
                    )
                    * self.nets[i][j](x)
                )
        return y


class Interface:
    def __init__(self, points1, points2, nr, outputs, i1, j1, i2, j2, overlap=0.2):
        x_min, x_max = min(points1[0], points2[0]), max(points1[0], points2[0])
        y_min, y_max = min(points1[1], points2[1]), max(points1[1], points2[1])
        self.geo = Rectangle(
            (x_min - overlap / 2, y_min - overlap / 2),
            (x_max + overlap / 2, y_max + overlap / 2),
        )
        self.nr = nr
        self.outputs = outputs
        self.i1 = i1
        self.j1 = j1
        self.i2 = i2
        self.j2 = j2

    def __call__(self, *args, **kwargs):
        points = self.geo.sample_boundary(self.nr)
        return points, {
            f"difference_{output}_{self.i1}_{self.j1}_{output}_{self.i2}_{self.j2}": 0
            for output in self.outputs
        }


class NetGridNode(NetNode):
    def __init__(
        self,
        inputs: Union[Tuple, List[str]],
        outputs: Union[Tuple, List[str]],
        x_segments: List[float] = None,
        y_segments: List[float] = None,
        z_segments: List[float] = None,
        t_segments: List[float] = None,
        columns: List[float] = None,
        rows: List[float] = None,
        *args,
        **kwargs,
    ):
        if columns is None:
            columns = []
        if rows is None:
            rows = []
        require_no_grad = False
        fixed = False
        self.columns = columns
        self.rows = rows
        self.main_net = NetEval(
            n_inputs=len(inputs),
            n_outputs=len(outputs),
            columns=columns,
            rows=rows,
            **kwargs,
        )
        super(NetGridNode, self).__init__(
            inputs, outputs, self.main_net, fixed, require_no_grad, *args, **kwargs
        )

    def get_grid(self, overlap, nr_points_per_interface_area=100):
        n_columns = self.main_net.n_columns
        n_rows = self.main_net.n_rows
        netnodes = []
        eqs = []
        constraints = []
        for i in range(n_columns):
            for j in range(n_rows):
                nn = NetNode(
                    inputs=self.inputs,
                    outputs=tuple(f"{output}_{i}_{j}" for output in self.outputs),
                    net=self.main_net.nets[i][j],
                    name=f"{self.name}[{i}][{j}]",
                )
                nn.is_reference = True
                netnodes.append(nn)
                if i > 0:
                    for output in self.outputs:
                        diff_Node = Difference(
                            f"{output}_{i - 1}_{j}",
                            f"{output}_{i}_{j}",
                            dim=2,
                            time=False,
                        )
                        eqs.append(diff_Node)

                    interface = Interface(
                        (self.columns[i], self.rows[j]),
                        (self.columns[i], self.rows[j + 1]),
                        nr_points_per_interface_area,
                        self.outputs,
                        i - 1,
                        j,
                        i,
                        j,
                        overlap=overlap,
                    )

                    constraints.append(
                        get_data_node(
                            interface, name=f"interface[{i - 1}][{j}]_[{i}][{j}]"
                        )
                    )
                if j > 0:
                    for output in self.outputs:
                        diff_Node = Difference(
                            f"{output}_{i}_{j - 1}",
                            f"{output}_{i}_{j}",
                            dim=2,
                            time=False,
                        )
                        eqs.append(diff_Node)

                    interface = Interface(
                        (self.columns[i], self.rows[j]),
                        (self.columns[i + 1], self.rows[j]),
                        nr_points_per_interface_area,
                        self.outputs,
                        i,
                        j - 1,
                        i,
                        j,
                        overlap=overlap,
                    )

                    constraints.append(
                        get_data_node(
                            interface, name=f"interface[{i}][{j - 1}]_[{i}][{j}]"
                        )
                    )
        return netnodes, eqs, constraints


def get_net_reg_grid_2d(
    inputs: Union[Tuple, List[str]],
    outputs: Union[Tuple, List[str]],
    name: str,
    columns: List[float],
    rows: List[float],
    **kwargs,
):
    if "overlap" in kwargs.keys():
        overlap = kwargs.pop("overlap")
    else:
        overlap = 0.2
    net = NetGridNode(
        inputs=inputs, outputs=outputs, columns=columns, rows=rows, name=name, **kwargs
    )
    nets, eqs, interfaces = net.get_grid(
        nr_points_per_interface_area=1000, overlap=overlap
    )
    nets.append(net)
    return nets, eqs, interfaces


def get_net_reg_grid(
    inputs: Union[Tuple, List[str]],
    outputs: Union[Tuple, List[str]],
    name: str,
    x_segments: List[float] = None,
    y_segments: List[float] = None,
    z_segments: List[float] = None,
    t_segments: List[float] = None,
    **kwargs,
):
    if "overlap" in kwargs.keys():
        overlap = kwargs.pop("overlap")
    else:
        overlap = 0.2
    net = NetGridNode(
        inputs=inputs,
        outputs=outputs,
        x_segments=x_segments,
        y_segments=y_segments,
        z_segments=z_segments,
        t_segments=t_segments,
        name=name,
        **kwargs,
    )
    nets, eqs, interfaces = net.get_grid(
        nr_points_per_interface_area=1000, overlap=overlap
    )
    nets.append(net)
    return nets, eqs, interfaces

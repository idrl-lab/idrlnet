"""Define variables, intermediate data format for the package."""

import torch
import itertools
from typing import List, Dict
import numpy as np
import os
from pyevtk.hl import pointsToVTK
import pathlib
import enum
from typing import Union
from collections import defaultdict
import pandas as pd
from idrlnet.header import DIFF_SYMBOL

__all__ = ["Loss", "Variables", "DomainVariables", "export_var"]


class Loss(enum.Enum):
    """Enumerate loss functions"""

    L1 = "L1"
    square = "square"


class LossFunction:
    """Manage loss functions"""

    @staticmethod
    def weighted_loss(variables, loss_function, name):
        if loss_function == Loss.L1.name or loss_function == Loss.L1:
            return LossFunction.weighted_L1_loss(variables, name=name)
        elif loss_function == Loss.square.name or loss_function == Loss.square:
            return LossFunction.weighted_square_loss(variables, name=name)
        raise NotImplementedError(f"loss function {loss_function} is not defined!")

    @staticmethod
    def weighted_L1_loss(variables: "Variables", name: str) -> "Variables":
        loss = 0.0
        for key, val in variables.items():
            if key.startswith("lambda_") or key == "area":
                continue
            elif "lambda_" + key in variables.keys():
                loss += torch.sum(
                    (torch.abs(val)) * variables["lambda_" + key] * variables["area"]
                )
            else:
                loss += torch.sum((torch.abs(val)) * variables["area"])
        return Variables({name: loss})

    @staticmethod
    def weighted_square_loss(variables: "Variables", name: str) -> "Variables":
        loss = 0.0
        for key, val in variables.items():
            if key.startswith("lambda_") or key == "area":
                continue
            elif "lambda_" + key in variables.keys():
                loss += torch.sum(
                    (val ** 2) * variables["lambda_" + key] * variables["area"]
                )
            else:
                loss += torch.sum((val ** 2) * variables["area"])
        return Variables({name: loss})


class Variables(dict):
    def __sub__(self, other: "Variables") -> "Variables":
        return Variables(
            {
                key: (self[key] if key in self else 0)
                - (other[key] if key in other else 0)
                for key in {**self, **other}
            }
        )

    def weighted_loss(self, name: str, loss_function: Union[Loss, str]) -> "Variables":
        """Regard the variable as residuals and reduce to a weighted_loss."""

        return LossFunction.weighted_loss(
            variables=self, loss_function=loss_function, name=name
        )

    def subset(self, subset_keys: List[str]) -> "Variables":
        """Construct a new variable with subset references"""

        return Variables({name: self[name] for name in subset_keys if name in self})

    def to_torch_tensor_(self) -> "Variables[str, torch.Tensor]":
        """Convert the variables to torch.Tensor"""

        for key, val in self.items():
            if not isinstance(val, torch.Tensor):
                self[key] = torch.Tensor(val)
                if (not key.startswith("lambda_")) and (not key == "area"):
                    self[key].requires_grad_()
        return self

    def to_ndarray_(self) -> "Variables[str, np.ndarray]":
        """convert to a numpy based variables"""

        for key, val in self.items():
            if isinstance(val, torch.Tensor):
                self[key] = val.detach().cpu().numpy()
        return self

    def to_ndarray(self) -> "Variables[str, np.ndarray]":
        """Return a new numpy based variables"""

        new_var = Variables()
        for key, val in self.items():
            if isinstance(val, torch.Tensor):
                new_var[key] = val.detach().cpu().numpy()
            else:
                new_var[key] = val
        return new_var

    def to_dataframe(self) -> pd.DataFrame:
        """merge to a pandas.DataFrame"""

        np_var = self.to_ndarray()
        keys, values = list(zip(*[(key, value) for key, value in np_var.items()]))
        values = np.concatenate([value for value in values], axis=-1)
        df = pd.DataFrame(data=values, columns=keys)
        return df

    def merge_tensor(self) -> torch.Tensor:
        """merge tensors in the Variable"""

        variable_list = [value for _, value in self.items()]
        variable_tensor = torch.cat(variable_list, dim=-1)
        return variable_tensor

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, variable_names: List[str]):
        """Construct Variables from torch.Tensor"""

        split_tensor = torch.split(tensor, 1, dim=-1)
        assert len(variable_names) == len(split_tensor)
        variables = cls()
        for name, var_t in zip(variable_names, split_tensor):
            variables[name] = var_t
        return variables

    def differentiate_one_step_(
        self: "Variables", independent_var: "Variables", required_derivatives: List[str]
    ):
        """One order of derivatives will be computed towards the required_derivatives."""

        required_derivatives = [d for d in required_derivatives if d not in self]
        required_derivatives_set = set(
            tuple(required_derivative.split(DIFF_SYMBOL))
            for required_derivative in required_derivatives
        )
        dependent_var_set = set(tuple(dv.split(DIFF_SYMBOL)) for dv in self.keys())
        computable_derivative_dict = defaultdict(set)
        for dv, rd in itertools.product(dependent_var_set, required_derivatives_set):
            if (
                len(rd) > len(dv)
                and rd[: len(dv)] == dv
                and rd[: len(dv) + 1] not in dependent_var_set
            ):
                computable_derivative_dict[rd[len(dv)]].add(DIFF_SYMBOL.join(dv))
        derivative_variables = Variables()
        for key, value in computable_derivative_dict.items():
            for v in value:
                f__x = torch.autograd.grad(
                    self[v],
                    independent_var[key],
                    grad_outputs=torch.ones_like(self[v]),
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True,
                )[0]
                if f__x is not None:
                    f__x.requires_grad_()
                else:
                    f__x = torch.zeros_like(self[v], requires_grad=True)
                derivative_variables[DIFF_SYMBOL.join([v, key])] = f__x
        self.update(derivative_variables)

    def differentiate_(
        self: "Variables", independent_var: "Variables", required_derivatives: List[str]
    ):
        """Derivatives will be computed towards the required_derivatives"""

        n_keys = 0
        new_keys = len(self.keys())
        while new_keys != n_keys:
            n_keys = new_keys
            self.differentiate_one_step_(independent_var, required_derivatives)
            new_keys = len(self.keys())

    @staticmethod
    def var_differentiate_one_step(
        dependent_var: "Variables",
        independent_var: "Variables",
        required_derivatives: List[str],
    ):
        """Perform one step of differentiate towards the required_derivatives"""

        dependent_var.differentiate_one_step_(independent_var, required_derivatives)

    def to_csv(self, filename: str) -> None:
        """Export variable to csv"""

        if not filename.endswith(".csv"):
            filename += ".csv"
        df = self.to_dataframe()
        df.to_csv(filename, index=False)

    def to_vtu(self, filename: str, coordinates=None) -> None:
        """Export variable to vtu"""

        coordinates = ["x", "y", "z"] if coordinates is None else coordinates
        shape = 0
        for axis in coordinates:
            if axis not in self.keys():
                self[axis] = np.zeros_like(next(iter(self.values())))
            else:
                shape = (len(self[axis]), 1)
        for key, value in self.items():
            if value.shape == (1, 1):
                self[key] = np.ones(shape) * value
            self[key] = np.asarray(self[key], dtype=np.float64)
        pointsToVTK(
            filename,
            self[coordinates[0]][:, 0].copy(),
            self[coordinates[1]][:, 0].copy(),
            self[coordinates[2]][:, 0].copy(),
            data={key: value[:, 0].copy() for key, value in self.items()},
        )

    def save(self, path, formats=None):
        """Export variable to various formats"""

        if formats is None:
            formats = ["np", "csv", "vtu"]
        np_var = self.to_ndarray()
        if "np" in formats:
            np.savez(path, **np_var)
        if "csv" in formats:
            np_var.to_csv(path)
        if "vtu" in formats:
            np_var.to_vtu(filename=path)

    @staticmethod
    def cat(*var_list) -> "Variables":
        """todo: catenate in var list"""
        return Variables()


DomainVariables = Dict[str, Variables]


def export_var(
    domain_var: DomainVariables, path="./inference_domain/results", formats=None
):
    """Export a dict of variables to ``csv``, ``vtu`` or ``npz``."""

    if formats is None:
        formats = ["csv", "vtu", "np"]
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True, parents=True)
    for key in domain_var.keys():
        domain_var[key].save(os.path.join(path, f"{key}"), formats)

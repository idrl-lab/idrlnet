"""Predefined equations

"""
from sympy import Function, Number, symbols

from idrlnet.pde import PdeNode

__all__ = [
    "DiffusionNode",
    "NavierStokesNode",
    "WaveNode",
    "BurgersNode",
    "SchrodingerNode",
    "AllenCahnNode",
]


def symbolize(s, input_variables=None):
    if type(s) in (list, tuple):
        return [symbolize(_s) for _s in s]
    elif type(s) is str:
        s = Function(s)(*input_variables)
    elif type(s) in [float, int]:
        s = Number(s)
    return s


class DiffusionNode(PdeNode):
    def __init__(self, T="T", D="D", Q=0, dim=3, time=True, **kwargs):
        super().__init__(**kwargs)
        self.T = T
        x, y, z, t = symbols("x y z t")
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        assert type(T) == str, "T should be string"

        T = symbolize(T, input_variables=input_variables)
        D = symbolize(D, input_variables=input_variables)
        Q = symbolize(Q, input_variables=input_variables)

        self.equations = {"diffusion_" + self.T: -Q}
        if time:
            self.equations["diffusion_" + self.T] += T.diff(t)
        coord = [x, y, z]
        for i in range(dim):
            s = coord[i]
            self.equations["diffusion_" + self.T] -= (D * T.diff(s)).diff(s)
        self.make_nodes()


class NavierStokesNode(PdeNode):
    def __init__(self, nu=0.1, rho=1.0, dim=2.0, time=False, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        assert self.dim in [2, 3], "dim should be 2 or 3"
        self.time = time
        x, y, z, t = symbols("x y z t")
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        u = symbolize("u", input_variables)
        v = symbolize("v", input_variables)
        w = symbolize("w", input_variables) if self.dim == 3 else Number(0)
        p = symbolize("p", input_variables)
        nu = symbolize(nu, input_variables)
        rho = symbolize(rho, input_variables)
        mu = rho * nu
        self.equations = {
            "continuity": rho.diff(t)
            + (rho * u).diff(x)
            + (rho * v).diff(y)
            + (rho * w).diff(z),
            "momentum_x": (
                (rho * u).diff(t)
                + (
                    u * ((rho * u).diff(x))
                    + v * ((rho * u).diff(y))
                    + w * ((rho * u).diff(z))
                )
                + p.diff(x)
                - (mu * u.diff(x)).diff(x)
                - (mu * u.diff(y)).diff(y)
                - (mu * u.diff(z)).diff(z)
            ),
            "momentum_y": (
                (rho * v).diff(t)
                + (
                    u * ((rho * v).diff(x))
                    + v * ((rho * v).diff(y))
                    + w * ((rho * v).diff(z))
                )
                + p.diff(y)
                - (mu * v.diff(x)).diff(x)
                - (mu * v.diff(y)).diff(y)
                - (mu * v.diff(z)).diff(z)
            ),
        }

        if self.dim == 3:
            self.equations["momentum_z"] = (
                (rho * w).diff(t)
                + (
                    u * ((rho * w).diff(x))
                    + v * ((rho * w).diff(y))
                    + w * ((rho * w).diff(z))
                )
                + p.diff(z)
                - (mu * w.diff(x)).diff(x)
                - (mu * w.diff(y)).diff(y)
                - (mu * w.diff(z)).diff(z)
            )
        self.make_nodes()


class WaveNode(PdeNode):
    def __init__(self, u="u", c="c", dim=3, time=True, **kwargs):
        super().__init__(**kwargs)
        self.u = u
        self.dim = dim
        self.time = time
        x, y, z, t = symbols("x y z t")
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        assert self.dim in [1, 2, 3], "dim should be 1, 2 or 3."
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")
        assert type(u) == str, "u should be string"
        u = symbolize(u, input_variables)
        c = symbolize(c, input_variables)
        self.equations = {
            "wave_equation": (
                u.diff(t, 2)
                - (c ** 2 * u.diff(x)).diff(x)
                - (c ** 2 * u.diff(y)).diff(y)
                - (c ** 2 * u.diff(z)).diff(z)
            )
        }
        self.make_nodes()


class BurgersNode(PdeNode):
    def __init__(self, u: str = "u", v="v"):
        super().__init__()
        x, t = symbols("x t")
        input_variables = {"x": x, "t": t}

        assert type(u) == str, "u needs to be string"
        u = symbolize(u, input_variables)
        v = symbolize(v, input_variables)

        self.equations = {
            f"burgers_{str(u)}": (u.diff(t) + u * u.diff(x) - v * (u.diff(x)).diff(x))
        }
        self.make_nodes()


class SchrodingerNode(PdeNode):
    def __init__(self, u="u", v="v", c=0.5):
        super().__init__()
        self.c = c
        x, t = symbols("x t")
        input_variables = {"x": x, "t": t}

        assert type(u) == str, "u should be string"
        u = symbolize(u, input_variables)

        assert type(v) == str, "v should be string"
        v = symbolize(v, input_variables)
        self.equations = {
            "real": u.diff(t) + self.c * v.diff(x, 2) + (u ** 2 + v ** 2) * v,
            "imaginary": v.diff(t) - self.c * u.diff(x, 2) - (u ** 2 + v ** 2) * u,
        }
        self.make_nodes()


class AllenCahnNode(PdeNode):
    def __init__(self, u="u", gamma_1=0.0001, gamma_2=5):
        super().__init__()
        self.gama_1 = gamma_1
        self.gama_2 = gamma_2
        x, t = symbols("x t")
        input_variables = {"x": x, "t": t}
        assert type(u) == str, "u should be string"
        u = symbolize(u, input_variables)
        self.equations = {
            "AllenCahn_"
            + str(u): u.diff(t)
            - self.gama_1 * u.diff(x, 2)
            - self.gama_2 * (u - u ** 3)
        }
        self.make_nodes()

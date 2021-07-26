from sympy import Symbol
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import idrlnet.shortcut as sc
import os
import torch

# parameter phase
L = 1.0

# define geometry
geo = sc.Line1D(-1.0, 1.0)

# define sympy varaibles to parametize domain curves
t_symbol = Symbol("t")
x = Symbol("x")
u = sp.Function("u")(x, t_symbol)
up = sp.Function("up")(x, t_symbol)
time_range = {t_symbol: (0, L)}


# constraint phase
@sc.datanode
class AllenInit(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        return geo.sample_interior(density=300, param_ranges={t_symbol: 0.0}), {
            "u": x ** 2 * sp.cos(sp.pi * x),
            "lambda_u": 100,
        }


@sc.datanode
class AllenBc(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        return geo.sample_boundary(
            density=200, sieve=sp.Eq(x, -1), param_ranges=time_range
        ), {
            "difference_u_up": 0,
            "difference_diff_u_diff_up": 0,
        }


@sc.datanode(name="allen_domain")
class AllenEq(sc.SampleDomain):
    def __init__(self):
        self.points = geo.sample_interior(
            density=2000, param_ranges=time_range, low_discrepancy=True
        )

    def sampling(self, *args, **kwargs):
        constraints = {"AllenCahn_u": 0}
        return self.points, constraints


@sc.datanode(name="data_evaluate")
class AllenPointsInference(sc.SampleDomain):
    def __init__(self):
        self.points = geo.sample_interior(
            density=5000, param_ranges=time_range, low_discrepancy=True
        )
        self.points = sc.Variables(self.points).to_torch_tensor_()
        self.constraints = {"AllenCahn_u": torch.zeros_like(self.points["x"])}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


@sc.datanode(name="re_sampling_domain")
class SpaceAdaptiveSampling(sc.SampleDomain):
    def __init__(self):
        self.points = geo.sample_interior(
            density=100, param_ranges=time_range, low_discrepancy=True
        )
        self.points = sc.Variables(self.points).to_torch_tensor_()
        self.constraints = {"AllenCahn_u": torch.zeros_like(self.points["x"])}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


@sc.datanode(name="allen_test")
def generate_plot_data():
    x = np.linspace(-1.0, 1.0, 100)
    t = np.linspace(0, 1.0, 100)
    x, t = np.meshgrid(x, t)
    points = sc.Variables(x=x.reshape(-1, 1), t=t.reshape(-1, 1))
    return points, {}


# computational node phase

net_u = sc.MLP([2, 128, 128, 128, 128, 2], activation=sc.Activation.tanh)
net_u = sc.NetNode(
    inputs=(
        "x",
        "t",
    ),
    outputs=("u",),
    name="net1",
    net=net_u,
)
xp = sc.ExpressionNode(name="xp", expression=x + 2)
get_tilde_u = sc.get_shared_net_node(
    net_u,
    inputs=(
        "xp",
        "t",
    ),
    outputs=("up",),
    name="net2",
    arch="mlp",
)

diff_u = sc.ExpressionNode(expression=u.diff(x), name="diff_u")
diff_up = sc.ExpressionNode(expression=up.diff(x), name="diff_up")

pde = sc.AllenCahnNode(u="u", gamma_1=0.0001, gamma_2=5)

boundary_up = sc.Difference(T="diff_u", S="diff_up")
boundary_u = sc.Difference(T="u", S="up")


# Receiver hook phase


class SpaceAdaptiveReceiver(sc.Receiver):
    def receive_notify(self, solver, message):
        if (
            sc.Signal.TRAIN_PIPE_END in message.keys()
            and solver.global_step % 1000 == 0
        ):
            sc.logger.info("space adaptive sampling...")
            results = solver.infer_step(
                {"data_evaluate": ["x", "t", "sdf", "AllenCahn_u"]}
            )
            residual_data = (
                results["data_evaluate"]["AllenCahn_u"].detach().cpu().numpy().ravel()
            )
            # sort the points by residual loss
            index = np.argsort(-1.0 * np.abs(residual_data))[:200]
            _points = {
                key: values[index].detach().cpu().numpy()
                for key, values in results["data_evaluate"].items()
            }
            _points.pop("AllenCahn_u")
            _points["area"] = np.zeros_like(_points["sdf"]) + (1.0 / 200)
            solver.set_domain_parameter("re_sampling_domain", {"points": _points})


class PostProcessReceiver(sc.Receiver):
    def __init__(self):
        if not os.path.exists("image"):
            os.mkdir("image")

    def receive_notify(self, solver, message):
        if (
            sc.Signal.TRAIN_PIPE_END in message.keys()
            and solver.global_step % 1000 == 1
        ):
            sc.logger.info("Post Processing...")
            points = s.infer_step({"allen_test": ["x", "t", "u"]})
            triang_total = tri.Triangulation(
                points["allen_test"]["t"].detach().cpu().numpy().ravel(),
                points["allen_test"]["x"].detach().cpu().numpy().ravel(),
            )
            plt.tricontourf(
                triang_total,
                points["allen_test"]["u"].detach().cpu().numpy().ravel(),
                100,
                vmin=-1,
                vmax=1,
            )
            tc_bar = plt.colorbar()
            tc_bar.ax.tick_params(labelsize=12)

            _points = solver.get_domain_parameter("re_sampling_domain", "points")
            if not isinstance(_points["t"], torch.Tensor):
                plt.scatter(_points["t"].ravel(), _points["x"].ravel(), marker="x", s=8)
            else:
                plt.scatter(
                    _points["t"].detach().cpu().numpy().ravel(),
                    _points["x"].detach().cpu().numpy().ravel(),
                    marker="x",
                    s=8,
                )

            plt.xlabel("$t$")
            plt.ylabel("$x$")
            plt.title("$u(x,t)$")
            plt.savefig(f"image/result_{solver.global_step}.png")
            plt.show()


# Solver phase
s = sc.Solver(
    sample_domains=(
        AllenInit(),
        AllenBc(),
        AllenEq(),
        AllenPointsInference(),
        SpaceAdaptiveSampling(),
        generate_plot_data(),
    ),
    netnodes=[net_u, get_tilde_u],
    pdes=[pde, xp, diff_up, diff_u, boundary_up, boundary_u],
    max_iter=60000,
    loading=True,
)

s.register_receiver(SpaceAdaptiveReceiver())
s.register_receiver(PostProcessReceiver())
s.solve()

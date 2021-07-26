import idrlnet.shortcut as sc
from math import pi
from sympy import Symbol
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

L = float(pi)

geo = sc.Line1D(0, L)
t_symbol = Symbol("t")
x = Symbol("x")
time_range = {t_symbol: (0, 2 * L)}
c = 1.54
external_filename = "external_sample.csv"


def generate_observed_data():
    if os.path.exists(external_filename):
        return
    points = geo.sample_interior(
        density=20, bounds={x: (0, L)}, param_ranges=time_range, low_discrepancy=True
    )
    points["u"] = np.sin(points["x"]) * (
        np.sin(c * points["t"]) + np.cos(c * points["t"])
    )
    points["u"][np.random.choice(len(points["u"]), 10, replace=False)] = 3.0
    points = {k: v.ravel() for k, v in points.items()}
    points = pd.DataFrame.from_dict(points)
    points.to_csv("external_sample.csv", index=False)


generate_observed_data()


# @sc.datanode(name='wave_domain')
@sc.datanode(name="wave_domain", loss_fn="L1")
class WaveExternal(sc.SampleDomain):
    def __init__(self):
        points = pd.read_csv("external_sample.csv")
        self.points = {
            col: points[col].to_numpy().reshape(-1, 1) for col in points.columns
        }
        self.constraints = {"u": self.points.pop("u")}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


@sc.datanode(name="wave_external")
class WaveEq(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        points = geo.sample_interior(
            density=1000, bounds={x: (0, L)}, param_ranges=time_range
        )
        constraints = {"wave_equation": 0.0}
        return points, constraints


@sc.datanode(name="center_infer")
class CenterInfer(sc.SampleDomain):
    def __init__(self):
        self.points = sc.Variables()
        self.points["t"] = np.linspace(0, 2 * L, 200).reshape(-1, 1)
        self.points["x"] = np.ones_like(self.points["t"]) * L / 2
        self.points["area"] = np.ones_like(self.points["t"])

    def sampling(self, *args, **kwargs):
        return self.points, {}


net = sc.get_net_node(
    inputs=(
        "x",
        "t",
    ),
    outputs=("u",),
    name="net1",
    arch=sc.Arch.mlp,
)
var_c = sc.get_net_node(inputs=("x",), outputs=("c",), arch=sc.Arch.single_var)
pde = sc.WaveNode(c="c", dim=1, time=True, u="u")
s = sc.Solver(
    sample_domains=(WaveExternal(), WaveEq()),
    netnodes=[net, var_c],
    pdes=[pde],
    # network_dir='square_network_dir',
    network_dir="network_dir",
    max_iter=5000,
)
s.solve()

_, ax = plt.subplots(1, 1, figsize=(8, 4))

coord = s.infer_step(domain_attr={"wave_domain": ["x", "t", "u"]})
num_t = coord["wave_domain"]["t"].cpu().detach().numpy().ravel()
num_u = coord["wave_domain"]["u"].cpu().detach().numpy().ravel()
ax.scatter(num_t, num_u, c="r", marker="o", label="predicted points")

print("true paratmeter c: {:.4f}".format(c))
predict_c = var_c.evaluate(torch.Tensor([[1.0]])).item()
print("predicted parameter c: {:.4f}".format(predict_c))

num_t = WaveExternal().sample_fn.points["t"].ravel()
num_u = WaveExternal().sample_fn.constraints["u"].ravel()
ax.scatter(num_t, num_u, c="b", marker="x", label="observed points")

s.sample_domains = (CenterInfer(),)
points = s.infer_step({"center_infer": ["t", "x", "u"]})
num_t = points["center_infer"]["t"].cpu().detach().numpy().ravel()
num_u = points["center_infer"]["u"].cpu().detach().numpy().ravel()
num_x = points["center_infer"]["x"].cpu().detach().numpy().ravel()
ax.plot(
    num_t, np.sin(num_x) * (np.sin(c * num_t) + np.cos(c * num_t)), c="k", label="exact"
)
ax.plot(num_t, num_u, "--", c="g", linewidth=4, label="predict")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("u")
# ax.set_title(f'Square loss ($x=0.5L$, c={predict_c:.4f}))')
ax.set_title(f"L1 loss ($x=0.5L$, c={predict_c:.4f})")
ax.grid(True)
ax.set_xlim([-0.5, 6.5])
ax.set_ylim([-3.5, 4.5])
# plt.savefig('square.png', dpi=1000, bbox_inches='tight', pad_inches=0.02)
plt.savefig("L1.png", dpi=1000, bbox_inches="tight", pad_inches=0.02)
plt.show()
plt.close()

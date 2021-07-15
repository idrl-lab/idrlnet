import idrlnet.shortcut as sc
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

x, y = sp.symbols("x y")
rec = sc.Rectangle((-1.0, -1.0), (1.0, 1.0))


@sc.datanode
class LeftRight(sc.SampleDomain):
    # Due to `name` is not specified, LeftRight will be the name of datanode automatically
    def sampling(self, *args, **kwargs):
        points = rec.sample_boundary(1000, sieve=((y > -1.0) & (y < 1.0)))
        constraints = {"T": 0.0}
        return points, constraints


@sc.datanode(name="up_down")
class UpDownBoundaryDomain(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        points = rec.sample_boundary(1000, sieve=((x > -1.0) & (x < 1.0)))
        constraints = {"normal_gradient_T": 0.0}
        return points, constraints


@sc.datanode(name="heat_domain")
class HeatDomain(sc.SampleDomain):
    def __init__(self):
        self.points = 1000

    def sampling(self, *args, **kwargs):
        points = rec.sample_interior(self.points)
        constraints = {"diffusion_T": 1.0}
        return points, constraints


net = sc.get_net_node(
    inputs=(
        "x",
        "y",
    ),
    outputs=("T",),
    name="net1",
    arch=sc.Arch.mlp,
)
pde = sc.DiffusionNode(T="T", D=1.0, Q=0.0, dim=2, time=False)
grad = sc.NormalGradient("T", dim=2, time=False)
s = sc.Solver(
    sample_domains=(HeatDomain(), LeftRight(), UpDownBoundaryDomain()),
    netnodes=[net],
    pdes=[pde, grad],
    max_iter=1000,
)
s.solve()

# Inference
s.set_domain_parameter("heat_domain", {"points": 10000})
coord = s.infer_step({"heat_domain": ["x", "y", "T"]})
num_x = coord["heat_domain"]["x"].cpu().detach().numpy().ravel()
num_y = coord["heat_domain"]["y"].cpu().detach().numpy().ravel()
num_Tp = coord["heat_domain"]["T"].cpu().detach().numpy().ravel()

# Ground truth
num_T = -num_x * num_x / 2 + 0.5

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
triang_total = tri.Triangulation(num_x, num_y)
ax[0].tricontourf(triang_total, num_Tp, 100, cmap="hot", vmin=0, vmax=0.5)
ax[0].axis("off")
ax[0].set_title("prediction")
ax[1].tricontourf(triang_total, num_T, 100, cmap="hot", vmin=0, vmax=0.5)
ax[1].axis("off")
ax[1].set_title("ground truth")
ax[2].tricontourf(
    triang_total, np.abs(num_T - num_Tp), 100, cmap="hot", vmin=0, vmax=0.5
)
ax[2].axis("off")
ax[2].set_title("absolute error")

plt.savefig("simple_poisson.png", dpi=300, bbox_inches="tight")

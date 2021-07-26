import idrlnet.shortcut as sc
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

x, y = sp.symbols("x y")
temp = sp.Symbol("temp")
temp_range = {temp: (-0.2, 0.2)}
rec = sc.Rectangle((-1.0, -1.0), (1.0, 1.0))


@sc.datanode
class Right(sc.SampleDomain):
    # Due to `name` is not specified, Right will be the name of datanode automatically
    def sampling(self, *args, **kwargs):
        points = rec.sample_boundary(
            1000, sieve=(sp.Eq(x, 1.0)), param_ranges=temp_range
        )
        constraints = sc.Variables({"T": 0.0})
        return points, constraints


@sc.datanode
class Left(sc.SampleDomain):
    # Due to `name` is not specified, Left will be the name of datanode automatically
    def sampling(self, *args, **kwargs):
        points = rec.sample_boundary(
            1000, sieve=(sp.Eq(x, -1.0)), param_ranges=temp_range
        )
        constraints = sc.Variables({"T": temp})
        return points, constraints


@sc.datanode(name="up_down")
class UpDownBoundaryDomain(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        points = rec.sample_boundary(
            1000, sieve=((x > -1.0) & (x < 1.0)), param_ranges=temp_range
        )
        constraints = sc.Variables({"normal_gradient_T": 0.0})
        return points, constraints


@sc.datanode(name="heat_domain")
class HeatDomain(sc.SampleDomain):
    def __init__(self):
        self.points = 1000

    def sampling(self, *args, **kwargs):
        points = rec.sample_interior(self.points, param_ranges=temp_range)
        constraints = sc.Variables({"diffusion_T": 1.0})
        return points, constraints


net = sc.get_net_node(
    inputs=("x", "y", "temp"), outputs=("T",), name="net1", arch=sc.Arch.mlp
)
pde = sc.DiffusionNode(T="T", D=1.0, Q=0.0, dim=2, time=False)
grad = sc.NormalGradient("T", dim=2, time=False)
s = sc.Solver(
    sample_domains=(HeatDomain(), Left(), Right(), UpDownBoundaryDomain()),
    netnodes=[net],
    pdes=[pde, grad],
    max_iter=3000,
)
s.solve()


def infer_temp(temp_num, file_suffix=None):
    temp_range[temp] = temp_num
    s.set_domain_parameter("heat_domain", {"points": 10000})
    coord = s.infer_step({"heat_domain": ["x", "y", "T"]})
    num_x = coord["heat_domain"]["x"].cpu().detach().numpy().ravel()
    num_y = coord["heat_domain"]["y"].cpu().detach().numpy().ravel()
    num_Tp = coord["heat_domain"]["T"].cpu().detach().numpy().ravel()

    # Ground truth
    num_T = -(num_x + 1 + temp_num) * (num_x - 1.0) / 2

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    triang_total = tri.Triangulation(num_x, num_y)
    ax[0].tricontourf(triang_total, num_Tp, 100, cmap="hot", vmin=-0.2, vmax=1.21 / 2)
    ax[0].axis("off")
    ax[0].set_title(f"prediction($T_l={temp_num:.2f}$)")
    ax[1].tricontourf(triang_total, num_T, 100, cmap="hot", vmin=-0.2, vmax=1.21 / 2)
    ax[1].axis("off")
    ax[1].set_title(f"ground truth($T_l={temp_num:.2f}$)")
    ax[2].tricontourf(
        triang_total, np.abs(num_T - num_Tp), 100, cmap="hot", vmin=0, vmax=1.21 / 2
    )
    ax[2].axis("off")
    ax[2].set_title("absolute error")
    if file_suffix is None:
        plt.savefig(f"poisson_{temp_num:.2f}.png", dpi=300, bbox_inches="tight")
        plt.show()
    else:
        plt.savefig(f"poisson_{file_suffix}.png", dpi=300, bbox_inches="tight")
        plt.show()


for i in range(41):
    temp_num = i / 100 - 0.2
    infer_temp(temp_num, file_suffix=i)

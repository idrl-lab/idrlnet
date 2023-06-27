import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import matplotlib.tri as tri
import idrlnet.shortcut as sc

x, y = sp.symbols("x y")
u = sp.Function("u")(x, y)
geo = sc.Rectangle((-1, -1), (1., 1.))


@sc.datanode(sigma=1000.0)
class Boundary(sc.SampleDomain):
    def __init__(self):
        self.points = geo.sample_boundary(100,)
        self.constraints = {"u": 0.}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


@sc.datanode(loss_fn="Identity")
class Interior(sc.SampleDomain):
    def __init__(self):
        self.points = geo.sample_interior(1000)
        self.constraints = {"integral_dxdy": 0,}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


def f(x, y):
    return 2 * sp.pi ** 2 * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)


dx_exp = sc.ExpressionNode(
    expression=0.5*(u.diff(x) ** 2 + u.diff(y) ** 2) - u * f(x, y), name="dxdy"
)
net = sc.get_net_node(inputs=("x", "y"), outputs=("u",), name="net", arch=sc.Arch.mlp)

integral = sc.ICNode("dxdy", dim=2, time=False)

s = sc.Solver(
    sample_domains=(Boundary(), Interior()),
    netnodes=[net],
    pdes=[
        dx_exp,
        integral,
    ],
    max_iter=10000,
)
s.solve()
coord = s.infer_step({"Interior": ["x", "y", "u"]})
num_x = coord["Interior"]["x"].cpu().detach().numpy().ravel()
num_y = coord["Interior"]["y"].cpu().detach().numpy().ravel()
num_Up = coord["Interior"]["u"].cpu().detach().numpy().ravel()

# Ground truth
num_U = np.sin(np.pi*num_x)*np.sin(np.pi*num_y)

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
triang_total = tri.Triangulation(num_x, num_y)
ax[0].tricontourf(triang_total, num_Up, 100, cmap="bwr", vmin=-1, vmax=1)
ax[0].axis("off")
ax[0].set_title("prediction")
ax[1].tricontourf(triang_total, num_U, 100, cmap="bwr", vmin=-1, vmax=1)
ax[1].axis("off")
ax[1].set_title("ground truth")
ax[2].tricontourf(
    triang_total, np.abs(num_U - num_Up), 100, cmap="bwr", vmin=0, vmax=0.5
)
ax[2].axis("off")
ax[2].set_title("absolute error")

plt.savefig("deepritz.png", dpi=300, bbox_inches="tight")

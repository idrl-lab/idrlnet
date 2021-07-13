import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import idrlnet.shortcut as sc

x = sp.symbols("x")
Line = sc.Line1D(0, 1)
y = sp.Function("y")(x)


@sc.datanode(name="interior")
class Interior(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        return Line.sample_interior(1000), {"dddd_y": 0}


@sc.datanode(name="left_boundary1")
class LeftBoundary1(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        return Line.sample_boundary(100, sieve=(sp.Eq(x, 0))), {"y": 0}


@sc.datanode(name="left_boundary2")
class LeftBoundary2(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        return Line.sample_boundary(100, sieve=(sp.Eq(x, 0))), {"d_y": 0}


@sc.datanode(name="right_boundary1")
class RightBoundary1(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        return Line.sample_boundary(100, sieve=(sp.Eq(x, 1))), {"dd_y": 0}


@sc.datanode(name="right_boundary2")
class RightBoundary2(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        return Line.sample_boundary(100, sieve=(sp.Eq(x, 1))), {"ddd_y": 0}


@sc.datanode(name="infer")
class Infer(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        return {"x": np.linspace(0, 1, 1000).reshape(-1, 1)}, {}


net = sc.get_net_node(inputs=("x",), outputs=("y",), name="net", arch=sc.Arch.mlp)

pde1 = sc.ExpressionNode(
    name="dddd_y", expression=y.diff(x).diff(x).diff(x).diff(x) + 1
)
pde2 = sc.ExpressionNode(name="d_y", expression=y.diff(x))
pde3 = sc.ExpressionNode(name="dd_y", expression=y.diff(x).diff(x))
pde4 = sc.ExpressionNode(name="ddd_y", expression=y.diff(x).diff(x).diff(x))

solver = sc.Solver(
    sample_domains=(
        Interior(),
        LeftBoundary1(),
        LeftBoundary2(),
        RightBoundary1(),
        RightBoundary2(),
    ),
    netnodes=[net],
    pdes=[pde1, pde2, pde3, pde4],
    max_iter=2000,
)
solver.solve()


# inference
def exact(x):
    return -(x ** 4) / 24 + x ** 3 / 6 - x ** 2 / 4


solver.sample_domains = (Infer(),)
points = solver.infer_step({"infer": ["x", "y"]})
xs = points["infer"]["x"].detach().cpu().numpy().ravel()
y_pred = points["infer"]["y"].detach().cpu().numpy().ravel()
plt.plot(xs, y_pred, label="Pred")
y_exact = exact(xs)
plt.plot(xs, y_exact, label="Exact", linestyle="--")
plt.legend()
plt.xlabel("x")
plt.ylabel("w")
plt.savefig("Euler_beam.png", dpi=300, bbox_inches="tight")
plt.show()

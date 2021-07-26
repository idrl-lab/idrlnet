import idrlnet.shortcut as sc
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x = sp.Symbol("x")
s = sp.Symbol("s")
f = sp.Function("f")(x)
geo = sc.Line1D(0, 5)


@sc.datanode
def interior():
    points = geo.sample_interior(1000)
    constraints = {"difference_lhs_rhs": 0}
    return points, constraints


@sc.datanode
def init():
    points = geo.sample_boundary(1, sieve=sp.Eq(x, 0))
    points["lambda_f"] = 1000 * np.ones_like(points["x"])
    constraints = {"f": 1}
    return points, constraints


@sc.datanode(name="InteriorInfer")
def infer():
    points = {"x": np.linspace(0, 5, 1000).reshape(-1, 1)}
    return points, {}


netnode = sc.get_net_node(inputs=("x",), outputs=("f",), name="net")
exp_lhs = sc.ExpressionNode(expression=f.diff(x) + f, name="lhs")

fs = sp.Symbol("fs")
exp_rhs = sc.Int1DNode(
    expression=sp.exp(s - x) * fs,
    var=s,
    lb=0,
    ub=x,
    expression_name="rhs",
    funs={"fs": {"eval": netnode, "input_map": {"x": "s"}, "output_map": {"f": "fs"}}},
    degree=10,
)
diff = sc.Difference(T="lhs", S="rhs", dim=1, time=False)

solver = sc.Solver(
    sample_domains=(interior(), init(), infer()),
    netnodes=[netnode],
    pdes=[exp_lhs, exp_rhs, diff],
    loading=True,
    max_iter=3000,
)
solver.solve()
points = solver.infer_step({"InteriorInfer": ["x", "f"]})
num_x = points["InteriorInfer"]["x"].detach().cpu().numpy().ravel()
num_f = points["InteriorInfer"]["f"].detach().cpu().numpy().ravel()

fig = plt.figure(figsize=(8, 4))
plt.plot(num_x, num_f)
plt.plot(num_x, np.exp(-num_x) * np.cosh(num_x))
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Prediction", "Exact"])
plt.savefig("ide.png", dpi=1000, bbox_inches="tight")
plt.show()

from sympy import Symbol, sin
import math
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import idrlnet.shortcut as sc

x = Symbol("x")
t_symbol = Symbol("t")
time_range = {t_symbol: (0, 1)}
geo = sc.Line1D(-1.0, 1.0)


@sc.datanode(name="burgers_equation")
def interior_domain():
    points = geo.sample_interior(
        10000, bounds={x: (-1.0, 1.0)}, param_ranges=time_range
    )
    constraints = {"burgers_u": 0}
    return points, constraints


@sc.datanode(name="t_boundary")
def init_domain():
    points = geo.sample_interior(100, param_ranges={t_symbol: 0.0})
    constraints = sc.Variables({"u": -sin(math.pi * x)})
    return points, constraints


@sc.datanode(name="x_boundary")
def boundary_domain():
    points = geo.sample_boundary(100, param_ranges=time_range)
    constraints = sc.Variables({"u": 0})
    return points, constraints


net = sc.get_net_node(
    inputs=(
        "x",
        "t",
    ),
    outputs=("u",),
    name="net1",
    arch=sc.Arch.mlp,
)
pde = sc.BurgersNode(u="u", v=0.01 / math.pi)
s = sc.Solver(
    sample_domains=(interior_domain(), init_domain(), boundary_domain()),
    netnodes=[net],
    pdes=[pde],
    max_iter=4000,
)
s.solve()

coord = s.infer_step(
    {
        "burgers_equation": ["x", "t", "u"],
        "t_boundary": ["x", "t"],
        "x_boundary": ["x", "t"],
    }
)
num_x = coord["burgers_equation"]["x"].cpu().detach().numpy().ravel()
num_t = coord["burgers_equation"]["t"].cpu().detach().numpy().ravel()
num_u = coord["burgers_equation"]["u"].cpu().detach().numpy().ravel()

init_x = coord["t_boundary"]["x"].cpu().detach().numpy().ravel()
init_t = coord["t_boundary"]["t"].cpu().detach().numpy().ravel()
boundary_x = coord["x_boundary"]["x"].cpu().detach().numpy().ravel()
boundary_t = coord["x_boundary"]["t"].cpu().detach().numpy().ravel()

triang_total = tri.Triangulation(num_t.flatten(), num_x.flatten())
u_pre = num_u.flatten()

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(221)
tcf = ax1.tricontourf(triang_total, u_pre, 100, cmap="jet")
tc_bar = plt.colorbar(tcf)
tc_bar.ax.tick_params(labelsize=10)
ax1.set_xlabel("$t$")
ax1.set_ylabel("$x$")
ax1.set_title("$u(x,t)$")
ax1.scatter(init_t, init_x, c="black", marker="x", s=8)
ax1.scatter(boundary_t, boundary_x, c="black", marker="x", s=8)
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.savefig("Burgers.png", dpi=500, bbox_inches="tight", pad_inches=0.02)

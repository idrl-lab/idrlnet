import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import idrlnet.shortcut as sc

x = sp.Symbol("x")
geo = sc.Line1D(-1, 0.5)


@sc.datanode(loss_fn="L1")
class Interior(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        points = geo.sample_interior(100)
        constraints = {
            "u": (np.cosh(0.5) - np.cosh(-1)) / 1.5 * (x + 1.0) + np.cosh(-1)
        }
        return points, constraints


net = sc.get_net_node(inputs=("x",), outputs=("u",), name="net", arch=sc.Arch.mlp)

s = sc.Solver(
    sample_domains=(Interior(),),
    netnodes=[net],
    pdes=[],
    network_dir="pretrain_network_dir",
    max_iter=1000,
)
s.solve()

points = s.infer_step({"Interior": ["x", "u"]})
num_x = points["Interior"]["x"].detach().cpu().numpy().ravel()
num_u = points["Interior"]["u"].detach().cpu().numpy().ravel()

xx = np.linspace(-1, 0.5, 1000, endpoint=True)
yy = np.cosh(xx)
plt.plot(xx, yy)
plt.plot(num_x, num_u)
plt.show()

from idrlnet import shortcut as sc
import sympy as sp
import torch
import numpy as np
from idrlnet.pde_op.equations import NavierStokesNode
from matplotlib import tri
import matplotlib.pyplot as plt

x, y = sp.symbols('x y')
rec = sc.Rectangle((-0.05, -0.05), (0.05, 0.05))


@sc.datanode(name='flow_domain')
class InteriorDomain(sc.SampleDomain):
    def __init__(self):
        points = rec.sample_interior(400000, bounds={x: (-0.05, 0.05), y: (-0.05, 0.05)})

        constraints = sc.Variables({'continuity': torch.zeros(len(points['x']), 1),
                                    'momentum_x': torch.zeros(len(points['x']), 1),
                                    'momentum_y': torch.zeros(len(points['x']), 1)})

        points['area'] = np.ones_like(points['area']) * 2.5e-6

        constraints['lambda_continuity'] = points['sdf']
        constraints['lambda_momentum_x'] = points['sdf']
        constraints['lambda_momentum_y'] = points['sdf']
        self.points = sc.Variables(points).to_torch_tensor_()
        self.constraints = sc.Variables(constraints).to_torch_tensor_()

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


@sc.datanode(name='left_right_down')
class LeftRightDownBoundaryDomain(sc.SampleDomain):
    def __init__(self):
        points = rec.sample_boundary(3333, sieve=(y < 0.05))
        constraints = sc.Variables({'u': torch.zeros(len(points['x']), 1), 'v': torch.zeros(len(points['x']), 1)})
        points['area'] = np.ones_like(points['area']) * 1e-4

        self.points = sc.Variables(points).to_torch_tensor_()
        self.constraints = sc.Variables(constraints).to_torch_tensor_()

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


@sc.datanode(name='up')
class UpBoundaryDomain(sc.SampleDomain):
    def __init__(self):
        points = rec.sample_boundary(10000, sieve=sp.Eq(y, 0.05))
        points['area'] = np.ones_like(points['area']) * 1e-4
        constraints = sc.Variables({'u': torch.ones(len(points['x']), 1), 'v': torch.zeros(len(points['x']), 1)})
        constraints['lambda_u'] = 1 - 20 * abs(points['x'].copy())
        self.points = sc.Variables(points).to_torch_tensor_()
        self.constraints = sc.Variables(constraints).to_torch_tensor_()

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


torch.autograd.set_detect_anomaly(True)
net = sc.MLP([2, 100, 100, 100, 100, 3], activation=sc.Activation.tanh, initialization=sc.Initializer.Xavier_uniform,
             weight_norm=False)

net_u = sc.NetNode(inputs=('x', 'y',), outputs=('u', 'v', 'p'), net=net, name='net_u')
pde = NavierStokesNode(nu=0.01, rho=1.0, dim=2, time=False)

s = sc.Solver(sample_domains=(InteriorDomain(), LeftRightDownBoundaryDomain(), UpBoundaryDomain()),
              netnodes=[net_u],
              pdes=[pde],
              max_iter=4000,
              network_dir='./result/tanh_Xavier_uniform',
              )
s.solve()


def interoir_domain_infer():
    points = rec.sample_interior(1000000, bounds={x: (-0.05, 0.05), y: (-0.05, 0.05)})
    constraints = sc.Variables({'continuity': torch.zeros(len(points['x']), 1),
                                'momentum_x': torch.zeros(len(points['x']), 1),
                                'momentum_y': torch.zeros(len(points['x']), 1)})
    return points, constraints


data_infer = sc.get_data_node(interoir_domain_infer, name='flow_domain')
s.sample_domains = [data_infer]

pred = s.infer_step({'flow_domain': ['x', 'y', 'v', 'u', 'p']})
num_x = pred['flow_domain']['x'].detach().cpu().numpy().ravel()
num_y = pred['flow_domain']['y'].detach().cpu().numpy().ravel()
num_u = pred['flow_domain']['u'].detach().cpu().numpy().ravel()
num_v = pred['flow_domain']['v'].detach().cpu().numpy().ravel()
num_p = pred['flow_domain']['p'].detach().cpu().numpy().ravel()
triang_total = tri.Triangulation(num_x, num_y)

triang_total = tri.Triangulation(num_x, num_y)
u_pre = num_u.flatten()

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)
tcf = ax1.tricontourf(triang_total, num_u, 100, cmap="jet")
tc_bar = plt.colorbar(tcf)
ax1.set_title('u')

ax2 = fig.add_subplot(132)
tcf = ax2.tricontourf(triang_total, num_v, 100, cmap="jet")
tc_bar = plt.colorbar(tcf)
ax2.set_title('v')

ax3 = fig.add_subplot(133)
tcf = ax3.tricontourf(triang_total, num_p, 100, cmap="jet")
tc_bar = plt.colorbar(tcf)
ax3.set_title('p')
plt.show()

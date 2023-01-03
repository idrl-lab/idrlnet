import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import idrlnet.shortcut as sc
from sympy import Symbol, sin
import pandas as pd
import torch
import matplotlib.tri as tri

x = Symbol('x')
y = Symbol('y')
t = Symbol('t')
geo = sc.Rectangle((1., -2.), (8., 2.))
u = sp.Function('u')(x, y, t)
v = sp.Function('v')(x, y, t)
p = sp.Function('p')(x, y, t)
time_range = {t: (0, 20)}
nu=0.01
rho=1

@sc.datanode(name='NS_domain', loss_fn='L1')
class NSExternal(sc.SampleDomain):
    def __init__(self):
        points = pd.read_csv('NSexternel_sample.csv')
        self.points = {col: points[col].to_numpy().reshape(-1, 1) for col in points.columns}
        self.constraints = {'u': self.points.pop('u'), 'v': self.points.pop('v'), 'p': self.points.pop('p')}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints

@sc.datanode(name='NS_external')
class NSEq(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        points = geo.sample_interior(density=2000, param_ranges=time_range)
        constraints = {'continuity': 0, 'momentum_x': 0, 'momentum_y': 0}
        return points, constraints

net = sc.MLP([3, 20, 20, 20, 20, 20, 20, 20, 20, 3], activation=sc.Activation.tanh)
net = sc.get_net_node(inputs=('x', 'y', 't'), outputs=('u', 'v', 'p'), name='net', arch=sc.Arch.mlp)
#var_nr = sc.get_net_node(inputs=('x', 'y'), outputs=('nu', 'rho'), arch=sc.Arch.single_var)
#pde = sc.NavierStokesNode(nu='nu', rho='rho', dim=2, time=True, u='u', v='v', p='p')
pde = sc.NavierStokesNode(nu=0.01, rho=1.0, dim=2, time=True)
s = sc.Solver(sample_domains=(NSExternal(), NSEq()),
              netnodes=[net],
              init_network_dirs=['network_dir_adam'],
              pdes=[pde],
              max_iter=100,
              opt_config=dict(optimizer='LBFGS', lr=1)
              )
#opt_config=dict(optimizer='LBFGS', lr=1)
# s = sc.Solver(sample_domains=(NSExternal(), NSEq()),
#               netnodes=[net, var_nr],
#               pdes=[pde],
#               network_dir='network_dir',
#               max_iter=10)
s.solve()


coord = s.infer_step(domain_attr={'NS_domain': ['x', 'y', 'u', 'v', 'p']})
num_xd = coord['NS_domain']['x'].cpu().detach().numpy().ravel()
num_yd = coord['NS_domain']['y'].cpu().detach().numpy().ravel()
num_ud = coord['NS_domain']['u'].cpu().detach().numpy().ravel()
num_vd = coord['NS_domain']['v'].cpu().detach().numpy().ravel()
num_pd = coord['NS_domain']['p'].cpu().detach().numpy().ravel()

# print("true paratmeter rho: {:.4f}".format(rho))
# predict_rho = var_nr.evaluate(torch.Tensor([[1.0]])).item()
# print("predicted parameter rho: {:.4f}".format(predict_rho))


points1 = pd.read_csv('NSexternel_test.csv')
points1 = {col: points1[col].to_numpy().reshape(-1, 1) for col in points1.columns}
x_test = torch.tensor(points1['x_test'].astype(np.float32))
y_test = torch.tensor(points1['y_test'].astype(np.float32))
t_test = torch.tensor(points1['t_test'].astype(np.float32))
u_test = torch.tensor(points1['u_test'].astype(np.float32))
v_test = torch.tensor(points1['v_test'].astype(np.float32))
p_test = torch.tensor(points1['p_test'].astype(np.float32))

U = s.netnodes[0].net(torch.cat([x_test, y_test, t_test], dim=1))

num_x = x_test.cpu().detach().numpy().ravel()
num_y = y_test.cpu().detach().numpy().ravel()
num_u = u_test.cpu().detach().numpy().ravel()
num_v = v_test.cpu().detach().numpy().ravel()
num_p = p_test.cpu().detach().numpy().ravel()

num_up = U[:, 0:1].cpu().detach().numpy().ravel()
num_vp = U[:, 1:2].cpu().detach().numpy().ravel()
num_pp = U[:, 2:3].cpu().detach().numpy().ravel()


triang_total = tri.Triangulation(num_x, num_y)

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 15,
         }

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111)
# ax.scatter(num_xi, num_yi, c='b', s=1, label='Domain')
# ax.set_xlabel('$x$', font2)
# ax.set_ylabel('$y$', font2)
# ax.set_title('collocation points', fontsize=18)
# plt.savefig('points.png', dpi=300, bbox_inches='tight')
# plt.show()

fig = plt.figure(figsize=(20, 4))
ax1 = fig.add_subplot(131)
tcf = ax1.tricontourf(triang_total, num_u, 100, cmap='jet')
tc_bar = plt.colorbar(tcf)
tc_bar.ax.tick_params(labelsize=12)
ax1.set_xlabel('$x$', font2)
ax1.set_ylabel('$y$', font2)
ax1.set_title('Exact $u$', fontsize=18)

ax2 = fig.add_subplot(132)
tcf = ax2.tricontourf(triang_total, num_up, 100, cmap='jet')
tc_bar = plt.colorbar(tcf)
tc_bar.ax.tick_params(labelsize=12)
ax2.set_xlabel('$x$', font2)
ax2.set_ylabel('$y$', font2)
ax2.set_title('Predicted $u$', fontsize=18)

ax3 = fig.add_subplot(133)
tcf = ax3.tricontourf(triang_total, num_u - num_up, 100, cmap='jet')
tc_bar = plt.colorbar(tcf)
tc_bar.ax.tick_params(labelsize=12)
ax3.set_xlabel('$x$', font2)
ax3.set_ylabel('$y$', font2)
ax3.set_title('Point-wise Error', fontsize=18)
plt.savefig('test_NS_u_c.png', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(20, 4))
ax1 = fig.add_subplot(131)
tcf = ax1.tricontourf(triang_total, num_v, 100, cmap='jet')
tc_bar = plt.colorbar(tcf)
tc_bar.ax.tick_params(labelsize=12)
ax1.set_xlabel('$x$', font2)
ax1.set_ylabel('$y$', font2)
ax1.set_title('Exact $v$', fontsize=18)

ax2 = fig.add_subplot(132)
tcf = ax2.tricontourf(triang_total, num_v, 100, cmap='jet')
tc_bar = plt.colorbar(tcf)
tc_bar.ax.tick_params(labelsize=12)
ax2.set_xlabel('$x$', font2)
ax2.set_ylabel('$y$', font2)
ax2.set_title('Predicted $v$', fontsize=18)

ax3 = fig.add_subplot(133)
tcf = ax3.tricontourf(triang_total, num_v - num_vp, 100, cmap='jet')
tc_bar = plt.colorbar(tcf)
tc_bar.ax.tick_params(labelsize=12)
ax3.set_xlabel('$x$', font2)
ax3.set_ylabel('$y$', font2)
ax3.set_title('Point-wise Error', fontsize=18)
plt.savefig('test_NS_v_c.png', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(20, 4))
ax1 = fig.add_subplot(131)
tcf = ax1.tricontourf(triang_total, num_p, 100, cmap='jet')
tc_bar = plt.colorbar(tcf)
tc_bar.ax.tick_params(labelsize=12)
ax1.set_xlabel('$x$', font2)
ax1.set_ylabel('$y$', font2)
ax1.set_title('Exact $p$', fontsize=18)

ax2 = fig.add_subplot(132)
tcf = ax2.tricontourf(triang_total, num_pp, 100, cmap='jet')
tc_bar = plt.colorbar(tcf)
tc_bar.ax.tick_params(labelsize=12)
ax2.set_xlabel('$x$', font2)
ax2.set_ylabel('$y$', font2)
ax2.set_title('Predicted $p$', fontsize=18)

ax3 = fig.add_subplot(133)
tcf = ax3.tricontourf(triang_total, num_p - num_pp, 100, cmap='jet')
tc_bar = plt.colorbar(tcf)
tc_bar.ax.tick_params(labelsize=12)
ax3.set_xlabel('$x$', font2)
ax3.set_ylabel('$y$', font2)
ax3.set_title('Point-wise Error', fontsize=18)
plt.savefig('test_NS_p_c.png', dpi=300, bbox_inches='tight')
plt.show()
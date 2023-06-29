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
rec = sc.Rectangle((0., 0.), (1.1, 0.41))
cir = sc.Circle((0.2, 0.2), 0.05)
geo = rec - cir
u = sp.Function('u')(x, y)
v = sp.Function('v')(x, y)
p = sp.Function('p')(x, y)
s11 = sp.Function('s11')(x, y)
s22 = sp.Function('s22')(x, y)
s12 = sp.Function('s12')(x, y)
nu=0.02
rho=1

@sc.datanode
class Inlet(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        points = rec.sample_boundary(1000, sieve=(sp.Eq(x, 0.)))
        constraints = sc.Variables({'u': 4 * (0.41 - y) * y / (0.41 * 0.41)})
        return points, constraints

@sc.datanode
class Outlet(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        points = geo.sample_boundary(1000, sieve=(sp.Eq(x, 1.1)))
        constraints = sc.Variables({'p': 0.})
        return points, constraints

@sc.datanode
class Wall(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        points = geo.sample_boundary(1000, sieve=((x > 0.) & (x < 1.1)))
        #print("points3", points)
        constraints = sc.Variables({'u': 0., 'v': 0.})
        return points, constraints

@sc.datanode(name='NS_external')
class Interior_domain(sc.SampleDomain):
    def __init__(self):
        self.density = 2000

    def sampling(self, *args, **kwargs):
        points = geo.sample_interior(2000)
        constraints = {'f_s11': 0., 'f_s22': 0., 'f_s12': 0., 'f_u': 0., 'f_v': 0., 'f_p': 0.}
        return points, constraints

@sc.datanode(name='NS_domain', loss_fn='L1')
class NSExternal(sc.SampleDomain):
    def __init__(self):
        points = pd.read_csv('NSexternel_sample.csv')
        self.points = {col: points[col].to_numpy().reshape(-1, 1) for col in points.columns}
        self.constraints = {'u': self.points.pop('u'), 'v': self.points.pop('v'), 'p': self.points.pop('p')}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints

net = sc.MLP([2, 40, 40, 40, 40, 40, 40, 40, 40, 6], activation=sc.Activation.tanh)
net = sc.get_net_node(inputs=('x', 'y'), outputs=('u', 'v', 'p', 's11', 's22', 's12'), name='net', arch=sc.Arch.mlp)
pde1 = sc.ExpressionNode(name='f_s11', expression=-p + 2 * nu * u.diff(x) - s11)
pde2 = sc.ExpressionNode(name='f_s22', expression=-p + 2 * nu * v.diff(y) - s22)
pde3 = sc.ExpressionNode(name='f_s12', expression=nu * (u.diff(y) + v.diff(x)) - s12)
pde4 = sc.ExpressionNode(name='f_u', expression=u * u.diff(x) + v * u.diff(y) - nu * (s11.diff(x) + s12.diff(y)))
pde5 = sc.ExpressionNode(name='f_v', expression=u * v.diff(x) + v * v.diff(y) - nu * (s12.diff(x) + s22.diff(y)))
pde6 = sc.ExpressionNode(name='f_p', expression=p + (s11 + s22) / 2)
s = sc.Solver(sample_domains=(Inlet(), Outlet(), Wall(), Interior_domain(), NSExternal()),
              netnodes=[net],
              init_network_dirs=['network_dir_adam'],
              pdes=[pde1, pde2, pde3, pde4, pde5, pde6],
              max_iter=300,
              opt_config = dict(optimizer='LBFGS', lr=1)
             )
#opt_config = dict(optimizer='LBFGS', lr=1)
#init_network_dirs=['network_dir_lbfgs'],
s.solve()

points1 = pd.read_csv('NSexternel_test.csv')
points1 = {col: points1[col].to_numpy().reshape(-1, 1) for col in points1.columns}
x_test = torch.tensor(points1['x_test'].astype(np.float32))
y_test = torch.tensor(points1['y_test'].astype(np.float32))
u_test = torch.tensor(points1['u_test'].astype(np.float32))
v_test = torch.tensor(points1['v_test'].astype(np.float32))
p_test = torch.tensor(points1['p_test'].astype(np.float32))

U = s.netnodes[0].net(torch.cat([x_test, y_test], dim=1))

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
plt.savefig('test_NS_u_Adam.png', dpi=300, bbox_inches='tight')
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
tcf = ax2.tricontourf(triang_total, num_vp, 100, cmap='jet')
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
plt.savefig('test_NS_v_Adam.png', dpi=300, bbox_inches='tight')
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
plt.savefig('test_NS_p_Adam.png', dpi=300, bbox_inches='tight')
plt.show()
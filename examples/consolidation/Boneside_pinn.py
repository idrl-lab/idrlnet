import idrlnet.shortcut as sc
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Aana import p_ratio_fn

cv = 0.6

x, y = sp.symbols('x y')
p = sp.Function('p')(x, y)
geo = sc.Rectangle((0, 0), (1, 1))


@sc.datanode
class Init(sc.SampleDomain):
    def __init__(self):
        self.points = geo.sample_boundary(100, sieve=(sp.Eq(y, 0.)))
        self.constraints = {'p': 1., 'lambda_p': 1 - np.power(self.points['x'], 5)}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


@sc.datanode
class Interior(sc.SampleDomain):
    def __init__(self):
        self.points = geo.sample_interior(10000)
        self.constraints = {'consolidation': np.zeros_like(self.points['x'])}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


@sc.datanode
class UpperBounds(sc.SampleDomain):
    def __init__(self):
        self.points = geo.sample_boundary(100, sieve=(sp.Eq(x, 1.)))
        self.constraints = {'p': 0.,
                            'lambda_p': np.power(self.points['y'], 0.2)}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


@sc.datanode
class LowerBounds(sc.SampleDomain):
    def __init__(self):
        self.points = geo.sample_boundary(100, sieve=(sp.Eq(x, 0.)))
        self.constraints = {'normal_gradient_p': 0., }

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


pde = sc.ExpressionNode(name='consolidation', expression=p.diff(y) - cv * p.diff(x, 2))
grad = sc.NormalGradient(T='p', dim=2, time=False)
net = sc.get_net_node(inputs=('x', 'y'), outputs=('p',), arch=sc.Arch.mlp, name='net')

s = sc.Solver(sample_domains=(Init(), Interior(), UpperBounds(), LowerBounds()),
              netnodes=[net],
              pdes=[pde, grad],
              max_iter=2000)
s.solve()


@sc.datanode
class MeshInterior(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        t_num = np.linspace(0.01, 1, 400, endpoint=True)
        z_num = np.linspace(0.01, 1, 100, endpoint=True)
        tt_num, zz_num = np.meshgrid(t_num, z_num)
        points = {'x': zz_num.reshape(-1, 1), 'y': tt_num.reshape(-1, 1)}
        return points, {}


s.sample_domains = (MeshInterior(),)
points = s.infer_step({'MeshInterior': ['x', 'y', 'p']})
x_num = points['MeshInterior']['x'].detach().cpu().numpy().ravel()
y_num = points['MeshInterior']['y'].detach().cpu().numpy().ravel()
p_num = points['MeshInterior']['p'].detach().cpu().numpy().ravel()

_, ax = plt.subplots(3, 1, figsize=(10, 10))

im = ax[0].scatter(y_num, x_num, c=p_num, vmin=0, vmax=1, cmap='jet')
ax[0].set_xlim([0, 1.01])
ax[0].set_ylim([-0.05, 1.05])
ax[0].set_ylabel('z(m)')
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='2%', pad=0.1)
plt.colorbar(im, cax=cax)
ax[0].set_title('Model Prediction: $p_{pred}=p/p_0(z,t)$')

ax[1].set_xlim([0, 1.01])
ax[1].set_ylim([-0.05, 1.05])
ax[1].set_ylabel('z(m)')
ax[1].set_title('Ground Truth: $p_{true}=p/p_0(z,t)$')
t_num = np.linspace(0.01, 1, 400, endpoint=True)
z_num = np.linspace(0.01, 1, 100, endpoint=True)
tt_num, zz_num = np.meshgrid(t_num, z_num)

p_ratio_num = p_ratio_fn(tt_num, zz_num)

im = ax[1].scatter(tt_num, zz_num, c=p_ratio_num, cmap='jet', vmin=0, vmax=1.)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='2%', pad=0.1)
plt.colorbar(im, cax=cax)

im = ax[2].scatter(tt_num, zz_num, c=p_num.reshape(100, 400) - p_ratio_num, cmap='bwr', vmin=-1,
                   vmax=1)
ax[2].set_xlim([0, 1.01])
ax[2].set_ylim([-0.05, 1.05])
ax[2].set_xlabel('t(yr)')
ax[2].set_ylabel('z(m)')
ax[2].set_title('Error: $p_{pred}-p_{true}$')

divider = make_axes_locatable(ax[2])
cax = divider.append_axes('right', size='2%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('test.png')
plt.show()
plt.close()

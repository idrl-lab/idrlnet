import idrlnet.shortcut as sc
import torch
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import abc
from matplotlib import tri

sc.use_gpu(device=0)

INTERVAL = 10000
DENSITY = 10000

r1, r2, r3, r4, r5, r6 = sp.symbols('r1 r2 r3 r4 r5 r6')

plate = sc.Tube2D((-1, -0.5), (1, 0.5))
hole_1 = sc.Circle(center=(-0.6, 0), radius=r1)
hole_2 = sc.Circle(center=(0., 0.), radius=r2)
hole_3 = sc.Circle(center=(0.5, -0.5), radius=r3)
hole_4 = sc.Circle(center=(0.5, 0.5), radius=r4)
hole_5 = sc.Circle(center=(-0.5, -0.5), radius=r5)
hole_6 = sc.Circle(center=(-0.5, 0.5), radius=r6)
geo = plate - hole_1 - hole_2 - hole_3 - hole_4 - hole_5 - hole_6

in_line = sc.Line((-1, -0.5), (-1, 0.5), normal=1)
out_line = sc.Line((1, -0.5), (1, 0.5), normal=1)
param_ranges = {r1: (0.05, 0.2),
                r2: (0.05, 0.2),
                r3: (0.05, 0.2),
                r4: (0.05, 0.2),
                r5: (0.05, 0.2),
                r6: (0.05, 0.2), }


class ReSampleDomain(sc.SampleDomain, metaclass=abc.ABCMeta):
    """
    Resampling collocated points every INTERVAL iterations.
    """
    count = 0
    points = sc.Variables()
    constraints = sc.Variables()

    def sampling(self, *args, **kwargs):
        if self.count % INTERVAL == 0:
            self.do_re_sample()
            sc.logger.info("Resampling...")
        self.count += 1
        return self.points, self.constraints

    @abc.abstractmethod
    def do_re_sample(self):
        pass


@sc.datanode
class InPlane(ReSampleDomain):
    def do_re_sample(self):
        self.points = sc.Variables(
            in_line.sample_boundary(param_ranges=param_ranges, density=DENSITY,
                                    low_discrepancy=True)).to_torch_tensor_()
        self.constraints = {'T': torch.ones_like(self.points['x'])}


@sc.datanode
class OutPlane(ReSampleDomain):
    def do_re_sample(self):
        self.points = sc.Variables(
            out_line.sample_boundary(param_ranges=param_ranges, density=DENSITY,
                                     low_discrepancy=True)).to_torch_tensor_()
        self.constraints = {'T': torch.zeros_like(self.points['x'])}


@sc.datanode(sigma=10.)
class Boundary(ReSampleDomain):
    def do_re_sample(self):
        self.points = sc.Variables(
            geo.sample_boundary(param_ranges=param_ranges, density=DENSITY, low_discrepancy=True)).to_torch_tensor_()
        self.constraints = {'normal_gradient_T': torch.zeros_like(self.points['x'])}


@sc.datanode
class Interior(ReSampleDomain):
    def do_re_sample(self):
        self.points = sc.Variables(
            geo.sample_interior(param_ranges=param_ranges, density=DENSITY, low_discrepancy=True)).to_torch_tensor_()
        self.constraints = {'diffusion_T': torch.zeros_like(self.points['x'])}


net = sc.get_net_node(inputs=('x', 'y', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6'), outputs=('T',), name='net',
                      arch=sc.Arch.mlp_xl)
pde = sc.DiffusionNode(T='T', D=1., Q=0, dim=2, time=False)
grad = sc.NormalGradient('T', dim=2, time=False)

s = sc.Solver(sample_domains=(InPlane(), OutPlane(), Boundary(), Interior()),
              netnodes=[net],
              pdes=[pde, grad],
              max_iter=100000,
              schedule_config=dict(scheduler='ExponentialLR', gamma=0.99998))

s.solve()


# Define inference domains.

@sc.datanode
class Inference(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        self.points = sc.Variables(
            geo.sample_interior(param_ranges=param_ranges, density=20000, low_discrepancy=True)).to_torch_tensor_()
        self.constraints = {'T__x': torch.zeros_like(self.points['x']),
                            'T__y': torch.zeros_like(self.points['x']), }
        return self.points, self.constraints


@sc.datanode
class BoundaryInference(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        self.points = sc.Variables(
            geo.sample_boundary(param_ranges=param_ranges, density=1000, low_discrepancy=True)).to_torch_tensor_()
        self.constraints = {'T__x': torch.zeros_like(self.points['x']),
                            'T__y': torch.zeros_like(self.points['x']), }
        return self.points, self.constraints


@sc.datanode
class InPlane(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        self.points = sc.Variables(
            in_line.sample_boundary(param_ranges=param_ranges, density=1000,
                                    low_discrepancy=True)).to_torch_tensor_()
        self.constraints = {'T__x': torch.zeros_like(self.points['x']),
                            'T__y': torch.zeros_like(self.points['x']), }
        return self.points, self.constraints


@sc.datanode
class OutPlane(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        self.points = sc.Variables(
            out_line.sample_boundary(param_ranges=param_ranges, density=1000,
                                     low_discrepancy=True)).to_torch_tensor_()
        self.constraints = {'T__x': torch.zeros_like(self.points['x']),
                            'T__y': torch.zeros_like(self.points['x']), }
        return self.points, self.constraints


s.sample_domains = (InPlane(), OutPlane(), Inference(), BoundaryInference())

count = [0]


def parameter_design(*args):
    """
    Do inference and plot the result.
    """

    param_ranges[r1] = args[0]
    param_ranges[r2] = args[1]
    param_ranges[r3] = args[2]
    param_ranges[r4] = args[3]
    param_ranges[r5] = args[4]
    param_ranges[r6] = args[5]

    points = s.infer_step({'Inference': ['x', 'y', 'T__x', 'T__y', ],
                           'BoundaryInference': ['x', 'y', 'T__x', 'T__y', ],
                           'InPlane': ['x', 'y', 'T__x', 'T__y', ],
                           'OutPlane': ['x', 'y', 'T__x', 'T__y', ]})

    plt.figure(figsize=(8, 4))
    fig = plt.gcf()
    fig.set_tight_layout(True)
    ########
    num_x = points['BoundaryInference']['x'].detach().cpu().numpy().ravel()
    num_y = points['BoundaryInference']['y'].detach().cpu().numpy().ravel()

    num_T__x = points['BoundaryInference']['T__x'].detach().cpu().numpy().ravel()
    num_T__y = points['BoundaryInference']['T__y'].detach().cpu().numpy().ravel()

    num_flux = np.sqrt(num_T__x ** 2 + num_T__y ** 2)
    plt.scatter(x=num_x, y=num_y, c=num_flux, s=3, vmin=0, vmax=0.8, cmap='bwr')

    ########
    num_x = points['InPlane']['x'].detach().cpu().numpy().ravel()
    num_y = points['InPlane']['y'].detach().cpu().numpy().ravel()

    num_T__x = points['InPlane']['T__x'].detach().cpu().numpy().ravel()
    num_T__y = points['InPlane']['T__y'].detach().cpu().numpy().ravel()

    num_flux = np.sqrt(num_T__x ** 2 + num_T__y ** 2)
    plt.scatter(x=num_x, y=num_y, c=num_flux, s=3, vmin=0, vmax=0.8, cmap='bwr')

    ########
    num_x = points['OutPlane']['x'].detach().cpu().numpy().ravel()
    num_y = points['OutPlane']['y'].detach().cpu().numpy().ravel()

    num_T__x = points['OutPlane']['T__x'].detach().cpu().numpy().ravel()
    num_T__y = points['OutPlane']['T__y'].detach().cpu().numpy().ravel()

    num_flux = np.sqrt(num_T__x ** 2 + num_T__y ** 2)
    plt.scatter(x=num_x, y=num_y, c=num_flux, s=3, vmin=0, vmax=0.8, cmap='bwr')

    ########
    num_x = points['Inference']['x'].detach().cpu().numpy().ravel()
    num_y = points['Inference']['y'].detach().cpu().numpy().ravel()

    num_T__x = points['Inference']['T__x'].detach().cpu().numpy().ravel()
    num_T__y = points['Inference']['T__y'].detach().cpu().numpy().ravel()

    num_flux = np.sqrt(num_T__x ** 2 + num_T__y ** 2)
    points['Inference']['T_flux'] = num_flux

    triang = tri.Triangulation(num_x, num_y)

    def apply_mask(triang, alpha=0.4):
        triangles = triang.triangles
        xtri = num_x[triangles] - np.roll(num_x[triangles], 1, axis=1)
        ytri = num_y[triangles] - np.roll(num_y[triangles], 1, axis=1)
        maxi = np.max(np.sqrt(xtri ** 2 + ytri ** 2), axis=1)
        triang.set_mask(maxi > alpha)

    apply_mask(triang, alpha=0.04)
    plt.tricontourf(triang, num_flux, 100, vmin=0, vmax=0.8, cmap='bwr')

    ax = plt.gca()
    ax.set_facecolor('k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-1, 1.])
    ax.set_ylim([-0.5, 0.5])
    plt.savefig("holes.png")
    plt.close()


parameter_design(0.14, 0.1, 0.2, 0.09, 0.05, 0.17)

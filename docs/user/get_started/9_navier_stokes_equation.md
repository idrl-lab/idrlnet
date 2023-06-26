# Navier-Stokes equations

This section repeats the Robust PINN method presented by [Peng et.al](https://deepai.org/publication/robust-regression-with-highly-corrupted-data-via-physics-informed-neural-networks).

## Steady 2D NS equations

The prototype problem of incompressible flow past a circular cylinder is considered.
![NS1](../../images/NS1.png)

The velocity vector is set to zero at all walls and the pressure is set to p = 0 at the outlet. The fluid density is taken as $\rho = 1kg/m^3$ and the dynamic viscosity is taken as $\mu = 2 · 10^{−2}kg/m^3$ . The velocity profile on the inlet is set as $u(0, y)=4 \frac{U_M}{H^2}(H-y) y$ with $U_M = 1m/s$ and $H = 0.41m$.

The two-dimensional steady-state Navier-Stokes equation is equivalently transformed into the following equations:

$$
\begin{equation}
\begin{aligned}
\sigma^{11} &=-p+2 \mu u_x \\
\sigma^{22} &=-p+2 \mu v_y \\
\sigma^{12} &=\mu\left(u_y+v_x\right) \\
p &=-\frac{1}{2}\left(\sigma^{11}+\sigma^{22}\right) \\
\left(u u_x+v u_y\right) &=\mu\left(\sigma_x^{11}+\sigma_y^{12}\right) \\
\left(u v_x+v v_y\right) &=\mu\left(\sigma_x^{12}+\sigma_y^{22}\right)
\end{aligned}
\end{equation}
$$

We construct a neural network with six outputs to satisfy the PDE constraints above:

$$
u, v, p, \sigma^{11}, \sigma^{12}, \sigma^{22}=\operatorname{net}(x, y)
$$

### Define Symbols and Geometric Objects

For the 2d problem, we define two coordinate symbols`x`and`y`, six variables$ u, v, p, \sigma^{11}, \sigma^{12}, \sigma^{22}$ are defined.

The geometry object is a simple rectangle and circle with the operator `-`.

```python
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
```

### Define Sampling Methods and Constraints

For the problem, three boundary conditions , PDE constraint and external data are presented. We use the robust-PINN model inspired by the traditional LAD (Least Absolute Derivation) approach, where the L1 loss replaces the squared L2 data loss.

```python
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
```

### Define Neural Networks and PDEs

In the PDE definition part, we add these PDE nodes:

```python
net = sc.MLP([2, 40, 40, 40, 40, 40, 40, 40, 40, 6], activation=sc.Activation.tanh)
net = sc.get_net_node(inputs=('x', 'y'), outputs=('u', 'v', 'p', 's11', 's22', 's12'), name='net', arch=sc.Arch.mlp)
pde1 = sc.ExpressionNode(name='f_s11', expression=-p + 2 * nu * u.diff(x) - s11)
pde2 = sc.ExpressionNode(name='f_s22', expression=-p + 2 * nu * v.diff(y) - s22)
pde3 = sc.ExpressionNode(name='f_s12', expression=nu * (u.diff(y) + v.diff(x)) - s12)
pde4 = sc.ExpressionNode(name='f_u', expression=u * u.diff(x) + v * u.diff(y) - nu * (s11.diff(x) + s12.diff(y)))
pde5 = sc.ExpressionNode(name='f_v', expression=u * v.diff(x) + v * v.diff(y) - nu * (s12.diff(x) + s22.diff(y)))
pde6 = sc.ExpressionNode(name='f_p', expression=p + (s11 + s22) / 2)
```

### Define A Solver

Direct use of Adam optimization is less effective, so the LBFGS optimization method or a combination of both (Adam+LBFGS) is used for training:

```python
s = sc.Solver(sample_domains=(Inlet(), Outlet(), Wall(), Interior_domain(), NSExternal()),
              netnodes=[net],
              init_network_dirs=['network_dir_adam'],
              pdes=[pde1, pde2, pde3, pde4, pde5, pde6],
              max_iter=300,
              opt_config = dict(optimizer='LBFGS', lr=1)
             )
```

The result is shown as follows:
![NS11](../../images/NS11.png)

## Unsteady 2D N-S equations with unknown parameters

A two-dimensional incompressible flow and dynamic vortex shedding past a circular cylinder in a steady-state are numerically simulated. Respectively, the Reynolds number of the incompressible flow is $Re = 100$. The kinematic viscosity of the fluid is $\nu = 0.01$. The cylinder diameter D is 1. The simulation domain size is
$[[-15,25] × [[-8,8]$. 选定计算域为$[1,8] × [-2,2]× [0,20]$.

![NS2](../../images/NS2.png)

$$
\begin{equation}
\begin{aligned}
&u_t+\lambda_1\left(u u_x+v u_y\right)=-p_x+\lambda_2\left(u_{x x}+u_{y y}\right) \\
&v_t+\lambda_1\left(u v_x+v v_y\right)=-p_y+\lambda_2\left(v_{x x}+v_{y y}\right)
\end{aligned}
\end{equation}
$$

where $\lambda_1$ and $\lambda_2$ are two unknown parameters to be recovered. We make the assumption that $u=\psi_y, \quad v=-\psi_x$

for some stream function $\psi(x, y)$. Under this assumption, the continuity equation will be automatically satisfied. The following architecture is used in this example,
$$
\psi, p=\operatorname{net}\left(t, x, y, \lambda_1, \lambda_2\right)
$$

### Define Symbols and Geometric Objects

We define three coordinate symbols `x`, `y` and `t`, three variables $u,v,p$ are defined.

```python
x = Symbol('x')
y = Symbol('y')
t = Symbol('t')
geo = sc.Rectangle((1., -2.), (8., 2.))
u = sp.Function('u')(x, y, t)
v = sp.Function('v')(x, y, t)
p = sp.Function('p')(x, y, t)
time_range = {t: (0, 20)}
```

### Define Sampling Methods and Constraints

This example has only two equation constraints, while the former has six equation constraints. We also use the LAD-PINN model. Then the PDE constrained optimization model is formulated as:

$$
\min _{\theta, \lambda} \frac{1}{\# \mathbf{D}_u} \sum_{\left(t_i, x_i, u_i\right) \in \mathbf{D}_u}\left|u_i-u_\theta\left(t_i, x_i ; \lambda\right)\right|+\omega \cdot L_{p d e} .
$$

```python
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
```

### Define Neural Networks and PDEs

IDRLnet defines a network node to represent the unknown Parameters.

```python
net = sc.MLP([3, 20, 20, 20, 20, 20, 20, 20, 20, 3], activation=sc.Activation.tanh)
net = sc.get_net_node(inputs=('x', 'y', 't'), outputs=('u', 'v', 'p'), name='net', arch=sc.Arch.mlp)
var_nr = sc.get_net_node(inputs=('x', 'y'), outputs=('nu', 'rho'), arch=sc.Arch.single_var)
pde = sc.NavierStokesNode(nu='nu', rho='rho', dim=2, time=True, u='u', v='v', p='p')
```

### Define A Solver

Two nodes trained together

```python
s = sc.Solver(sample_domains=(NSExternal(), NSEq()),
              netnodes=[net, var_nr],
              pdes=[pde],
              network_dir='network_dir',
              max_iter=10000)
```

Finally, the real velocity field and pressure field at t=10s are compared with the predicted results:
![NS22](../../images/NS22.png)
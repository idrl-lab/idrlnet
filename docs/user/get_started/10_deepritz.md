# Deepritz

This section repeats the Deepritz method presented by [Weinan E and Bing Yu](https://link.springer.com/article/10.1007/s40304-018-0127-z).

Consider the 2d Poisson's equation such as the following:

$$
\begin{equation}
\begin{aligned}
-\Delta u=f, & \text { in } \Omega \\
u=0, & \text { on } \partial \Omega
\end{aligned}
\end{equation}
$$

Based on the scattering theorem, its weak form is that both sides are multiplied by$ v \in H_0^1$(which can be interpreted as some function bounded by 0),to get

$$
\int f v=-\int v \Delta u=(\nabla u, \nabla v)
$$

The above equation holds for any $v \in H_0^1$. The bilinear part of the right-hand side of the equation with respect to $u,v$ is symmetric and yields the bilinear term:

$$
a(u, v)=\int \nabla u \cdot \nabla v
$$

By the Poincaré inequality, the $a(\cdot, \cdot)$ is a positive definite operator. By positive definite, we mean that there exists $\alpha >0$, such that

$$
a(u, u) \geq \alpha\|u\|^2, \quad \forall u \in H_0^1
$$

The remaining term is a linear generalization of $v$, which is $l(v)$, which yields the equation:

$$
a(u, v) = l(v)
$$

For this equation, by discretizing $u,v$ in the same finite dimensional subspace, we can obtain a symmetric positive definite system of equations, which is the family of Galerkin methods, or we can transform it into a polarization problem to solve it.

To find $u$ satisfies

$$
a(u, v) = l(v), \quad \forall v \in H_0^1
$$

For a symmetric positive definite $a$ , which is equivalent to solving the variational minimization problem, that is, finding $u$, such that holds, where

$$
J(u) = \frac{1}{2} a(u, u) - l(u)
$$

Specifically

$$
\min _{u \in H_0^1} J(u)=\frac{1}{2} \int\|\nabla u\|_2^2-\int f v
$$

The DeepRitz method is similar to the PINN approach, replacing the neural network with u, and after sampling the region, just solve it with a solver like Adam. Written as

$$
\begin{equation}
\min _{\left.\hat{u}\right|_{\partial \Omega}=0} \hat{J}(\hat{u})=\frac{1}{2} \frac{S_{\Omega}}{N_{\Omega}} \sum\left\|\nabla \hat{u}\left(x_i, y_i\right)\right\|_2^2-\frac{S_{\Omega}}{N_{\partial \Omega}} \sum f\left(x_i, y_i\right) \hat{u}\left(x_i, y_i\right)
\end{equation}
$$

Note that the original $u \in H_0^1$, which is zero on the boundary, is transformed into an unconstrained problem by adding the penalty function term:

$$
\begin{equation}
\begin{gathered}
\min \hat{J}(\hat{u})=\frac{1}{2} \frac{S_{\Omega}}{N_{\Omega}} \sum\left\|\nabla \hat{u}\left(x_i, y_i\right)\right\|_2^2-\frac{S_{\Omega}}{N_{\Omega}} \sum f\left(x_i, y_i\right) \hat{u}\left(x_i, y_i\right)+\beta \frac{S_{\partial \Omega}}{N_{\partial \Omega}} \\
\sum \hat{u}^2\left(x_i, y_i\right)
\end{gathered}
\end{equation}
$$

Consider the 2d Poisson's equation defined on $\Omega=[-1,1]\times[-1,1]$, which satisfies $f=2 \pi^2 \sin (\pi x) \sin (\pi y)$.

### Define Sampling Methods and Constraints

For the problem, boundary condition and PDE constraint are presented and use the Identity loss.

```python
@sc.datanode(sigma=1000.0)
class Boundary(sc.SampleDomain):
    def __init__(self):
        self.points = geo.sample_boundary(100,)
        self.constraints = {"u": 0.}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


@sc.datanode(loss_fn="Identity")
class Interior(sc.SampleDomain):
    def __init__(self):
        self.points = geo.sample_interior(1000)
        self.constraints = {"integral_dxdy": 0,}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints
```

### Define Neural Networks and PDEs

In the PDE definition section, based on the DeepRitz method we add two types of PDE nodes:

```python
def f(x, y):
    return 2 * sp.pi ** 2 * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)

dx_exp = sc.ExpressionNode(
    expression=0.5*(u.diff(x) ** 2 + u.diff(y) ** 2) - u * f(x, y), name="dxdy"
)
net = sc.get_net_node(inputs=("x", "y"), outputs=("u",), name="net", arch=sc.Arch.mlp)

integral = sc.ICNode("dxdy", dim=2, time=False)
```

The result is shown as follows:
![deepritz](https://github.com/xiangzixuebit/picture/raw/3d73005f3642f10400975659479e856fb99f6518/deepritz.png)

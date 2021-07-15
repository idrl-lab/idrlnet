import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import sympy as sp
from typing import Dict
import pickle
import math

import idrlnet.shortcut as sc

x = sp.Symbol("x")
u = sp.Function("u")(x)
geo = sc.Line1D(-1, 0.5)


@sc.datanode(sigma=1000.0)
class Boundary(sc.SampleDomain):
    def __init__(self):
        self.points = geo.sample_boundary(
            1,
        )
        self.constraints = {"u": np.cosh(self.points["x"])}

    def sampling(self, *args, **kwargs):
        return self.points, self.constraints


@sc.datanode(loss_fn="L1")
class Interior(sc.SampleDomain):
    def sampling(self, *args, **kwargs):
        points = geo.sample_interior(10000)
        constraints = {
            "integral_dx": 0,
        }
        return points, constraints


@sc.datanode
class InteriorInfer(sc.SampleDomain):
    def __init__(self):
        self.points = sc.Variables()
        self.points["x"] = np.linspace(-1, 0.5, 1001, endpoint=True).reshape(-1, 1)
        self.points["area"] = np.ones_like(self.points["x"])

    def sampling(self, *args, **kwargs):
        return self.points, {}


# plot Intermediate results
class PlotReceiver(sc.Receiver):
    def __init__(self):
        if not os.path.exists("plot"):
            os.mkdir("plot")
        xx = np.linspace(-1, 0.5, 1001, endpoint=True)
        self.xx = xx
        angle = np.linspace(0, math.pi * 2, 100)
        yy = np.cosh(xx)

        xx_mesh, angle_mesh = np.meshgrid(xx, angle)
        yy_mesh = yy * np.cos(angle_mesh)
        zz_mesh = yy * np.sin(angle_mesh)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca(projection="3d")
        ax.set_zlim3d(-1.25 - 1, 0.75 + 1)
        ax.set_ylim3d(-2, 2)
        ax.set_xlim3d(-2, 2)

        my_col = cm.cool((yy * np.ones_like(angle_mesh) - 1.0) / 0.6)
        ax.plot_surface(yy_mesh, zz_mesh, xx_mesh, facecolors=my_col)
        ax.view_init(elev=15.0, azim=0)
        ax.dist = 5
        plt.axis("off")
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(f"plot/p_exact.png")
        plt.show()
        plt.close()
        self.predict_history = []

    def receive_notify(self, obj: sc.Solver, message: Dict):
        if sc.Signal.SOLVE_START in message or (
            sc.Signal.TRAIN_PIPE_END in message and obj.global_step % 200 == 0
        ):
            print("plotting")
            points = s.infer_step({"InteriorInfer": ["x", "u"]})
            num_x = points["InteriorInfer"]["x"].detach().cpu().numpy().ravel()
            num_u = points["InteriorInfer"]["u"].detach().cpu().numpy().ravel()
            angle = np.linspace(0, math.pi * 2, 100)

            xx_mesh, angle_mesh = np.meshgrid(num_x, angle)
            yy_mesh = num_u * np.cos(angle_mesh)
            zz_mesh = num_u * np.sin(angle_mesh)

            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca(projection="3d")
            ax.set_zlim3d(-1.25 - 1, 0.75 + 1)
            ax.set_ylim3d(-2, 2)
            ax.set_xlim3d(-2, 2)

            my_col = cm.cool((num_u * np.ones_like(angle_mesh) - 1.0) / 0.6)
            ax.plot_surface(yy_mesh, zz_mesh, xx_mesh, facecolors=my_col)
            ax.view_init(elev=15.0, azim=0)
            ax.dist = 5
            plt.axis("off")
            plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
            plt.savefig(f"plot/p_{obj.global_step}.png")
            plt.show()
            plt.close()

            self.predict_history.append((num_u, obj.global_step))
        if sc.Signal.SOLVE_END in message:
            try:
                with open("result.pickle", "rb") as f:
                    self.predict_history = pickle.load(f)
            except:
                with open("result.pickle", "wb") as f:
                    pickle.dump(self.predict_history, f)
            for yy, step in self.predict_history:
                if step == 0:
                    plt.plot(yy, self.xx, label=f"iter={step}")
                if step == 200:
                    plt.plot(yy, self.xx, label=f"iter={step}")
                if step == 800:
                    plt.plot(yy[::100], self.xx[::100], "-o", label=f"iter={step}")
            plt.plot(np.cosh(self.xx)[::100], self.xx[::100], "-x", label="exact")
            plt.plot([0, np.cosh(-1)], [-1, -1], "--", color="gray")
            plt.plot([0, np.cosh(0.5)], [0.5, 0.5], "--", color="gray")
            plt.legend()
            plt.xlim([0, 1.7])
            plt.xlabel("y")
            plt.ylabel("x")
            plt.savefig("iterations.png")
            plt.show()
            plt.close()


dx_exp = sc.ExpressionNode(
    expression=sp.Abs(u) * sp.sqrt((u.diff(x)) ** 2 + 1), name="dx"
)
net = sc.get_net_node(inputs=("x",), outputs=("u",), name="net", arch=sc.Arch.mlp)

integral = sc.ICNode("dx", dim=1, time=False)

s = sc.Solver(
    sample_domains=(Boundary(), Interior(), InteriorInfer()),
    netnodes=[net],
    init_network_dirs=["pretrain_network_dir"],
    pdes=[
        dx_exp,
        integral,
    ],
    max_iter=1500,
)
s.register_receiver(PlotReceiver())
s.solve()

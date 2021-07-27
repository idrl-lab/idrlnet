import idrlnet.shortcut as sc
import math
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

order = 10
s = 0
h = 1
cv = 0.6
x, t = sp.symbols('x t')

for k in range(1, order + 1):
    s += (-1) ** (k - 1) / (2 * k - 1) * sp.cos((2 * k - 1) * math.pi * x / 2 / h) * sp.exp(
        -(2 * k - 1) ** 2 * math.pi ** 2 / 4 * cv * t / h / h)
p_ratio = s * 4 / math.pi
p_ratio_fn = sc.lambdify_np(p_ratio, ['t', 'x'])

if __name__ == '__main__':
    t_num = np.linspace(0.01, 1, 100, endpoint=True)
    z_num = np.linspace(0.01, 1, 100, endpoint=True)
    tt_num, zz_num = np.meshgrid(t_num, z_num)
    p_ratio_num = p_ratio_fn(tt_num, zz_num)
    plt.scatter(tt_num, zz_num, c=p_ratio_num, cmap='jet', vmin=0, vmax=1)
    plt.show()

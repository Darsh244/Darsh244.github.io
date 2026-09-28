import math
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 + 5

def golden_section_search(x0, y0, f, a, b, tolerance=1e-7):
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    def dist_sq(x): return (x - x0)**2 + (f(x) - y0)**2
    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)
    while abs(b - a) > tolerance:
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            f_x2 = f_x1
            x1 = a + resphi * (b - a)
            f_x1 = dist_sq(x1)
        else:
            a = x1
            x1 = x2
            f_x1 = f_x2
            x2 = b - resphi * (b - a)
            f_x2 = dist_sq(x2)
    best_x = (a + b) / 2
    return math.sqrt(dist_sq(best_x)), best_x

def golden_steps(x0, y0, f, a, b, n_steps=15):
    """Run several Golden Section steps and record the bracket at each one."""
    resphi = 2 - (1 + math.sqrt(5)) / 2
    D = lambda x: (x - x0)**2 + (f(x) - y0)**2
    steps = []
    for _ in range(n_steps):
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)
        steps.append((a, b, x1, x2))
        if D(x1) < D(x2):
            b = x2
        else:
            a = x1
    return steps

def plot_brackets(x0, y0, steps, x_star):
    """One row per iteration: the bracket [a, b] shrinking around x1, x2."""
    n = len(steps)
    plt.figure(figsize=(7, 0.35 * n + 1.5))
    for k, (a, b, x1, x2) in enumerate(steps):
        row = n - k  # iteration 1 at the top, later iterations below it
        plt.plot([a, b], [row, row], "b|-", markersize=10, linewidth=2,
                 label="bracket [a, b]" if k == 0 else None)
        plt.plot([x1, x2], [row, row], "ks", markersize=5,
                 label="x1, x2" if k == 0 else None)
    plt.axvline(x_star, color="green", linestyle="--",
                label=f"x* = {x_star:.4f}")
    plt.yticks(range(1, n + 1), [f"Iteration {n - r + 1}" for r in range(1, n + 1)])
    plt.title(f"Golden section brackets for point ({x0}, {y0})")
    plt.xlabel("x")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"media/golden_brackets_{x0}_{y0}.png", dpi=150)
    plt.close()

def plot_distance(x0, y0, x_star):
    """Full plot: the point, the curve, and the shortest distance."""
    d = np.hypot(x_star - x0, f(x_star) - y0)
    x = np.linspace(min(x0, -1) - 1, max(x0, 1) + 1, 400)
    plt.figure()
    plt.plot(x, f(x), "b-", label="y = x^2 + 5")
    plt.plot(x0, y0, "ro", label=f"point ({x0}, {y0})")
    plt.plot(x_star, f(x_star), "g*", markersize=12,
             label=f"closest point ({x_star:.4f}, {f(x_star):.4f})")
    plt.plot([x0, x_star], [y0, f(x_star)], "g--", label=f"d = {d:.2f}")
    plt.title(f"Shortest distance from ({x0}, {y0})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-1, 15)
    plt.legend()
    plt.savefig(f"media/golden_distance_{x0}_{y0}.png", dpi=150)
    plt.close()

for x0, y0 in [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]:
    d, x_star = golden_section_search(x0, y0, f, -2, 2)
    steps = golden_steps(x0, y0, f, -2, 2)
    plot_brackets(x0, y0, steps, x_star)
    plot_distance(x0, y0, x_star)
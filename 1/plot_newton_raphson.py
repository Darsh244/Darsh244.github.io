import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 + 5
df = lambda x: 2 * x
ddf = lambda x: 2

def newton_steps(x0, y0, x=0.0, n_steps=3):
    """Run a few Newton-Raphson steps on D'(x) = 0 and return every x."""
    xs = [x]
    for _ in range(n_steps):
        D_prime = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        D_double_prime = 2 + 2 * df(x)**2 + 2 * (f(x) - y0) * ddf(x)
        x = x - D_prime / D_double_prime
        xs.append(x)
    return xs

def plot_iterates(x0, y0, steps, x_star):
    """Zoomed plot: each iterate as a colored dot, shrinking and darkening
    as it approaches x* (shown as a green ring)."""
    lo, hi = min(steps), max(steps)
    pad = max(0.15 * (hi - lo), 0.5)  # keep some curve visible if all steps match
    x = np.linspace(lo - pad, hi + pad, 200)

    plt.figure(figsize=(8, 5))
    plt.plot(x, f(x), "b-", label="y = x^2 + 5", zorder=1)
    plt.plot(x_star, f(x_star), "o", markersize=18, markerfacecolor="none",
             markeredgecolor="green", markeredgewidth=2,
             label=f"x* = {x_star:.4f}", zorder=2)

    # color each iterate distinctly
    cmap = plt.cm.autumn_r(np.linspace(0.2, 0.9, len(steps)))
    for k, xk in enumerate(steps):
        size = 12 - 1.8 * k
        plt.plot(xk, f(xk), "o", color=cmap[k], markersize=size,
                  markeredgecolor="black", markeredgewidth=0.6,
                  label=f"x{k} = {xk:.4f}", zorder=3)

    plt.title(f"Newton-Raphson iterates for point ({x0}, {y0})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"media/iterates_{x0}_{y0}.png", dpi=150)
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
    plt.savefig(f"media/distance_{x0}_{y0}.png", dpi=150)
    plt.close()

for x0, y0 in [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]:
    steps = newton_steps(x0, y0)
    plot_distance(x0, y0, steps[-1])
    if steps[0] != steps[-1]:
        plot_iterates(x0, y0, steps, steps[-1])
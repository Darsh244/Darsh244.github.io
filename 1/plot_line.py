import numpy as np
import matplotlib.pyplot as plt

pts = [(0, 0.5), (2, 3.5), (1, 1.5), (3, 7.5)]
xs = np.array([p[0] for p in pts])
ys = np.array([p[1] for p in pts])

# Analytical fit
c0_star, c1_star = -0.2, 2.3


def plot_analytical_fit(xs, ys, c0_star, c1_star):
    x_plot = np.linspace(-0.5, 3.5, 100)

    plt.figure()
    plt.scatter(xs, ys, color="red", zorder=5, label="data points")
    plt.plot(
        x_plot,
        c0_star + c1_star * x_plot,
        "b-",
        label=f"y* = {c0_star} + {c1_star}x"
    )

    plt.title("Analytical line fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("media/line_analytical_fit.png", dpi=150)
    plt.close()


# Derivatives
def dPdc0(c0, c1):
    return 8 * c0 + 12 * c1 - 26


def dPdc1(c0, c1):
    return 12 * c0 + 28 * c1 - 62


# Newton-Raphson algorithm
def line_newton(n_iter=15):
    c0, c1 = 0.0, 0.0
    history = [(c0, c1)]

    for _ in range(n_iter):
        c0 = c0 - dPdc0(c0, c1) / 8
        c1 = c1 - dPdc1(c0, c1) / 28

        history.append((c0, c1))

    return history


# Newton-Raphson plotting
def plot_newton_convergence(history, c0_star, c1_star):
    iters = range(len(history))
    c0_vals = [h[0] for h in history]
    c1_vals = [h[1] for h in history]

    plt.figure()
    plt.plot(iters, c0_vals, "o-", label="c0")
    plt.plot(iters, c1_vals, "s-", label="c1")

    plt.axhline(
        c0_star, color="C0", linestyle="--",
        linewidth=1, label=f"c0* = {c0_star}"
    )
    plt.axhline(
        c1_star, color="C1", linestyle="--",
        linewidth=1, label=f"c1* = {c1_star}"
    )

    plt.title("Newton-Raphson convergence of c0 and c1")
    plt.xlabel("iteration")
    plt.ylabel("value")
    plt.legend()
    plt.savefig("media/line_newton_convergence.png", dpi=150)
    plt.close()


# Generate plots
plot_analytical_fit(xs, ys, c0_star, c1_star)

history = line_newton()
plot_newton_convergence(history, c0_star, c1_star)
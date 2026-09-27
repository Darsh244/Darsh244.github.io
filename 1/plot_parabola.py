import numpy as np
import matplotlib.pyplot as plt

pts = [(0, 0.5), (2, 3.5), (1, 1.5), (3, 7.5)]
xs = np.array([p[0] for p in pts])
ys = np.array([p[1] for p in pts])

# Analytical fit
c0_star, c1_star, c2_star = 0.55, 0.05, 0.75


def plot_analytical_fit(xs, ys, c0_star, c1_star, c2_star):
    x_plot = np.linspace(-0.5, 3.5, 100)

    plt.figure()
    plt.scatter(xs, ys, color="red", zorder=5, label="data points")
    plt.plot(
        x_plot,
        c0_star + c1_star * x_plot + c2_star * x_plot**2,
        "b-",
        label=f"y* = {c0_star} + {c1_star}x + {c2_star}x^2"
    )

    plt.title("Analytical parabola fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("media/parabola_analytical_fit.png", dpi=150)
    plt.close()


# Derivatives
def dPdc0(c0, c1, c2):
    return 8 * c0 + 12 * c1 + 28 * c2 - 26


def dPdc1(c0, c1, c2):
    return 12 * c0 + 28 * c1 + 72 * c2 - 62


def dPdc2(c0, c1, c2):
    return 28 * c0 + 72 * c1 + 196 * c2 - 166


# Newton-Raphson algorithm
def parabola_newton(n_iter=300):
    c0, c1, c2 = 0.0, 0.0, 0.0
    history = [(c0, c1, c2)]

    for _ in range(n_iter):
        c0 = c0 - dPdc0(c0, c1, c2) / 8
        c1 = c1 - dPdc1(c0, c1, c2) / 28
        c2 = c2 - dPdc2(c0, c1, c2) / 196

        history.append((c0, c1, c2))

    return history


# Newton-Raphson plotting
def plot_newton_convergence(history, c0_star, c1_star, c2_star):
    iters = range(len(history))
    c0_vals = [h[0] for h in history]
    c1_vals = [h[1] for h in history]
    c2_vals = [h[2] for h in history]

    plt.figure()
    plt.plot(iters, c0_vals, "-", label="c0")
    plt.plot(iters, c1_vals, "-", label="c1")
    plt.plot(iters, c2_vals, "-", label="c2")

    plt.axhline(
        c0_star, color="C0", linestyle="--",
        linewidth=1, label=f"c0* = {c0_star}"
    )
    plt.axhline(
        c1_star, color="C1", linestyle="--",
        linewidth=1, label=f"c1* = {c1_star}"
    )
    plt.axhline(
        c2_star, color="C2", linestyle="--",
        linewidth=1, label=f"c2* = {c2_star}"
    )

    plt.title("Newton-Raphson convergence of c0, c1 and c2")
    plt.xlabel("iteration")
    plt.ylabel("value")
    plt.legend()
    plt.savefig("media/parabola_newton_convergence.png", dpi=150)
    plt.close()


# Generate plots
plot_analytical_fit(xs, ys, c0_star, c1_star, c2_star)

history = parabola_newton()
plot_newton_convergence(history, c0_star, c1_star, c2_star)
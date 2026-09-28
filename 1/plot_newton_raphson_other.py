import numpy as np
import matplotlib.pyplot as plt

def find_distance_newton(x0, y0, f, df, ddf, initial_guess=0.0,
                         tolerance=1e-7, max_iter=100):
    x = initial_guess
    for _ in range(max_iter):
        # D'(x)
        D_prime = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        # D''(x)
        D_double_prime = 2 + 2 * (df(x)**2) + 2 * (f(x) - y0) * ddf(x)
        # Newton-Raphson update step
        next_x = x - D_prime / D_double_prime
        if abs(next_x - x) < tolerance:
            x = next_x
            break
        x = next_x
    shortest_distance = ((x - x0)**2 + (f(x) - y0)**2)**0.5
    return shortest_distance, x

def newton_history(x0, y0, f, df, ddf, initial_guess=0.0,
                    tolerance=1e-7, max_iter=100):
    """Same iteration as find_distance_newton, but returns every x visited."""
    x = initial_guess
    history = [x]
    for _ in range(max_iter):
        D_prime = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        D_double_prime = 2 + 2 * (df(x)**2) + 2 * (f(x) - y0) * ddf(x)
        next_x = x - D_prime / D_double_prime
        history.append(next_x)
        if abs(next_x - x) < tolerance:
            break
        x = next_x
    return history

def plot_iterates(name, label, f, x0, y0, steps, x_star):
    """Zoomed plot: each iterate as a colored dot, shrinking and darkening
    as it approaches x* (shown as a green ring). Same style as the main
    Newton-Raphson section."""
    steps = steps[:4] 
    lo, hi = min(steps), max(steps)
    pad = max(0.15 * (hi - lo), 0.5)
    x = np.linspace(lo - pad, hi + pad, 200)

    plt.figure(figsize=(8, 5))
    plt.plot(x, f(x), "b-", label=f"y = {label}", zorder=1)
    plt.plot(x_star, f(x_star), "o", markersize=18, markerfacecolor="none",
             markeredgecolor="green", markeredgewidth=2,
             label=f"x* = {x_star:.4f}", zorder=2)

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
    plt.savefig(f"media/iterates_{name}_{x0}_{y0}.png", dpi=150)
    plt.close()

def run_case(name, label, f, df, ddf, guess, points, xlim, ylim):
    """Find the shortest distance for each point, plot the results, and
    plot the iterate convergence for each point separately."""
    x = np.linspace(*xlim, 400)
    plt.figure()
    plt.plot(x, f(x), "b-", label=f"y = {label}")

    for x0, y0 in points:
        d, x_star = find_distance_newton(x0, y0, f, df, ddf, initial_guess=guess)
        y_star = f(x_star)
        print(f"{label:10s} ({x0}, {y0}): x* = {x_star:.4f}, "
              f"closest = ({x_star:.4f}, {y_star:.4f}), d = {d:.4f}")

        plt.plot(x0, y0, "ro")                        # the query point
        plt.plot(x_star, y_star, "g*", markersize=12)  # closest point on curve
        plt.plot([x0, x_star], [y0, y_star], "g--",
                 label=f"({x0}, {y0}): d = {d:.2f}")

        steps = newton_history(x0, y0, f, df, ddf, initial_guess=guess)
        plot_iterates(name, label, f, x0, y0, steps, x_star)

    plt.title(f"Shortest distance to y = {label}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.gca().set_aspect("equal")
    plt.legend(loc="best")
    plt.savefig(f"media/{name}.png", dpi=150)
    plt.close()

# other polynomial: f(x) = x^3/4 - x
run_case("cubic", "x^3/4 - x",
         lambda x: x**3 / 4 - x, lambda x: 3 * x**2 / 4 - 1, lambda x: 3 * x / 2,
         0.0, [(4, 0), (-4, 2)], (-5, 5), (-5, 5))

# exponential: f(x) = e^x
run_case("exp", "e^x", np.exp, np.exp, np.exp,
         0.0, [(2, 0), (0, -2)], (-3, 4), (-3, 6))

# radical: f(x) = sqrt(x)
run_case("sqrt", "sqrt(x)",
         np.sqrt, lambda x: 0.5 / np.sqrt(x), lambda x: -0.25 * x**-1.5,
         1.0, [(4, 0), (2, 3)], (0, 8), (-1, 4))
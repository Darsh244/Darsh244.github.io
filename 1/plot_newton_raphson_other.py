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

def run_case(name, label, f, df, ddf, guess, points, xlim, ylim):
    """Find the shortest distance for each point and plot the results."""
    x = np.linspace(*xlim, 400)
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
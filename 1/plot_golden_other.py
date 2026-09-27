import math
import numpy as np
import matplotlib.pyplot as plt

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

def run_case(name, label, f, a, b, points, xlim, ylim):
    """Search [a, b] for each point and plot the results."""
    x = np.linspace(a, b, 2000)
    plt.plot(x, f(x), "b-", label=f"y = {label}")

    for x0, y0 in points:
        d, x_star = golden_section_search(x0, y0, f, a, b)
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
    plt.savefig(f"media/golden_{name}.png", dpi=150)
    plt.close()

# other polynomial: f(x) = x^4/4 - x^2, searched on [-4, 4]
run_case("quartic", "x^4/4 - x^2", lambda x: x**4 / 4 - x**2,
         -4, 4, [(3, 0), (-3, -1)], (-4, 4), (-4, 6))

# logarithmic: f(x) = ln(x), searched on [0.05, 10] (needs x > 0)
run_case("log", "ln(x)", np.log,
         0.05, 10, [(2, 3), (5, -1)], (0, 10), (-3, 4))

# variable in the denominator: f(x) = 1/x, searched on [0.05, 10] (needs x > 0)
run_case("reciprocal", "1/x", lambda x: 1 / x,
         0.05, 10, [(2, 0), (1, 3)], (0, 6), (-1, 6))
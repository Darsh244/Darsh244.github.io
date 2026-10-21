import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from regression import *

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})
Xtr, ttr, Xte, tte = load_data()
out = {}

# (a)
tr, te, W = polynomial_regression(Xtr, ttr, Xte, tte)
deg = np.arange(1, 8)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(deg, tr, "o-", label="Training RMS error")
ax.semilogy(deg, te, "s-", label="Test RMS error")
ax.set_xlabel("Polynomial degree"); ax.set_ylabel("RMS error ($1000s, log scale)")
ax.set_xticks(deg); ax.grid(True, which="both", alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig("media/poly_error_vs_degree.png"); plt.close(fig)
out["a"] = {"train": tr, "test": te}

# (b)
w = W[1]
out["b"] = {"bias": w[0], "weights": dict(zip(FEATURES, w[1:])),
            "rm_mu": Xtr[:, 5].mean(), "rm_sd": Xtr[:, 5].std(),
            "dis_mu": Xtr[:, 7].mean(), "dis_sd": Xtr[:, 7].std()}

# (c)
out["c"] = {}
for d in (1, 3, 7):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    r = polynomial_regression_1d_vis(Xtr, ttr, Xte, tte, d, feature=2, ax=ax)
    ax.set_title(f"Degree {d}: train RMS {r[0]:.2f}, test RMS {r[1]:.2f}")
    fig.tight_layout(); fig.savefig(f"media/poly_1d_degree{d}.png"); plt.close(fig)
    out["c"][d] = r

# (d)
cv = polynomial_regression_reg(Xtr, ttr, seed=0)
lams = [l for l in LAMBDAS if l > 0]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogx(lams, [cv[l] for l in lams], "o-", label="Mean validation RMS error")
ax.axhline(cv[0], color="tab:red", ls="--", label=r"$\lambda = 0$ baseline")
best = min(lams, key=lambda l: cv[l])
ax.plot(best, cv[best], "k*", ms=14, label=rf"best $\lambda = {best}$")
ax.set_xlabel(r"Regularization constant $\lambda$"); ax.set_ylabel("Mean validation RMS error ($1000s)")
ax.grid(True, which="both", alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig("media/poly_reg_cv.png"); plt.close(fig)
out["d"] = {str(k): v for k, v in cv.items()}

res = np.array([[polynomial_regression_reg(Xtr, ttr, seed=s)[l] for l in LAMBDAS] for s in range(50)])
out["d_rob"] = {"median": np.median(res, 0).tolist(), "wins": np.bincount(res.argmin(1), minlength=7).tolist()}
json.dump(out, open("results.json", "w"), indent=1, default=float)
print(json.dumps(out["d_rob"]), best)

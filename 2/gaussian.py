import numpy as np
import matplotlib.pyplot as plt

from regression import (load_data, normalize, fit_ls, fit_ridge, rmse,
                        LAMBDAS)


def median_width(X):
    """Width heuristic: median Euclidean distance between training points."""
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return np.median(dist[np.triu_indices(len(X), k=1)])


def gaussian_design(X, centers, width):
    """Bias column, then exp(-||x - mu_j||^2 / (2 width^2)) for each center."""
    sq = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.column_stack([np.ones(len(X)),
                            np.exp(-sq / (2 * width ** 2))])


def gaussian_regression(X_train, t_train, X_test, t_test,
                        ks=range(5, 100, 10), seed=0):
    """Gaussian basis regression with K random training points as centers."""
    Xn_train, Xn_test = normalize(X_train, X_test)
    width = median_width(Xn_train)
    perm = np.random.default_rng(seed).permutation(len(Xn_train))
    train_err, test_err = [], []
    for K in ks:
        centers = Xn_train[perm[:K]]          
        Phi_train = gaussian_design(Xn_train, centers, width)
        Phi_test = gaussian_design(Xn_test, centers, width)
        w = fit_ls(Phi_train, t_train)
        train_err.append(rmse(Phi_train, w, t_train))
        test_err.append(rmse(Phi_test, w, t_test))
    return list(ks), train_err, test_err, width


def gaussian_regression_reg(X_train, t_train, n_basis=90, lambdas=LAMBDAS,
                            k=10, seed=0):
    """10-fold CV for L2-regularized Gaussian basis regression (all features).

    Normalization, width and centers are all computed from the training part
    of each fold only (the validation points are never used to build them).
    """
    n = len(X_train)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    cv_err = {lam: [] for lam in lambdas}
    for i in range(k):
        val = folds[i]
        trn = np.concatenate([folds[j] for j in range(k) if j != i])
        Xn_trn, Xn_val = normalize(X_train[trn], X_train[val])
        width = median_width(Xn_trn)
        perm = rng.permutation(len(trn))
        centers = Xn_trn[perm[:n_basis]]
        Phi_trn = gaussian_design(Xn_trn, centers, width)
        Phi_val = gaussian_design(Xn_val, centers, width)
        for lam in lambdas:
            w = fit_ridge(Phi_trn, t_train[trn], lam)
            cv_err[lam].append(rmse(Phi_val, w, t_train[val]))
    return {lam: float(np.mean(e)) for lam, e in cv_err.items()}


if __name__ == "__main__":
    plt.rcParams.update({"figure.dpi": 150, "font.size": 11})
    Xtr, ttr, Xte, tte = load_data()

    ks, tr, te, width = gaussian_regression(Xtr, ttr, Xte, tte)
    print(f"width = {width:.3f}")
    for K, a, b in zip(ks, tr, te):
        print(f"K={K:3d}  train RMS {a:8.3f}  test RMS {b:10.3f}")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(ks, tr, "o-", label="Training RMS error")
    ax.semilogy(ks, te, "s-", label="Test RMS error")
    ax.set_xlabel("Number of Gaussian basis functions $K$")
    ax.set_ylabel("RMS error ($1000s, log scale)")
    ax.set_xticks(ks); ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("media/gaussian_error_vs_K.png")
    plt.close(fig)

    cv = gaussian_regression_reg(Xtr, ttr)
    for lam in LAMBDAS:
        print(f"lambda={lam:<6}  mean validation RMS {cv[lam]:.3f}")
    lams = [l for l in LAMBDAS if l > 0]
    best = min(lams, key=lambda l: cv[l])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(lams, [cv[l] for l in lams], "o-",
                label="Mean validation RMS error")
    ax.axhline(cv[0], color="tab:red", ls="--", label=r"$\lambda = 0$ baseline")
    ax.plot(best, cv[best], "k*", ms=14, label=rf"best $\lambda = {best}$")
    ax.set_xlabel(r"Regularization constant $\lambda$")
    ax.set_ylabel("Mean validation RMS error ($1000s)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("media/gaussian_reg_cv.png")
    plt.close(fig)

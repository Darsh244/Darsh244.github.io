import numpy as np
import csv
import matplotlib.pyplot as plt

N_TRAIN = 100
FEATURES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
            "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]


def load_data(path="BostonHousing.csv"):
    """First 100 rows -> training set, remaining 406 rows -> test set."""
    with open(path) as f:
        rows = list(csv.reader(f))[1:]         
    data = np.array(rows, dtype=float)
    X, t = data[:, :-1], data[:, -1]
    return X[:N_TRAIN], t[:N_TRAIN], X[N_TRAIN:], t[N_TRAIN:]


def normalize(X_train, *others):
    """Z-score using the training mean/std only (std of 0 is replaced by 1)."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return [(X_train - mu) / sd] + [(X - mu) / sd for X in others]


def poly_design(X, degree):
    """Bias column, then x1..x1^d, x2..x2^d, ... (monomials, no cross-terms)."""
    cols = [np.ones(len(X))]
    for j in range(X.shape[1]):
        for p in range(1, degree + 1):
            cols.append(X[:, j] ** p)
    return np.column_stack(cols)


def fit_ls(Phi, t):
    """Unregularized least squares, w = pinv(Phi) t."""
    return np.linalg.lstsq(Phi, t, rcond=None)[0]


def fit_ridge(Phi, t, lam):
    """L2-regularized least squares; the bias weight is not penalized."""
    P = np.eye(Phi.shape[1])
    P[0, 0] = 0.0
    return np.linalg.solve(Phi.T @ Phi + lam * P, Phi.T @ t)


def rmse(Phi, w, t):
    return np.sqrt(np.mean((Phi @ w - t) ** 2))


def polynomial_regression(X_train, t_train, X_test, t_test, max_degree=7):
    """Fit degree 1..max_degree polynomial basis regression (all features)."""
    Xn_train, Xn_test = normalize(X_train, X_test)
    train_err, test_err, weights = [], [], {}
    for d in range(1, max_degree + 1):
        Phi_train = poly_design(Xn_train, d)
        Phi_test = poly_design(Xn_test, d)
        w = fit_ls(Phi_train, t_train)
        train_err.append(rmse(Phi_train, w, t_train))
        test_err.append(rmse(Phi_test, w, t_test))
        weights[d] = w
    return train_err, test_err, weights


def polynomial_regression_1d_vis(X_train, t_train, X_test, t_test,
                                 degree, feature=2, ax=None):
    """Fit a degree-d polynomial to a single feature and plot it."""
    x_train, x_test = X_train[:, [feature]], X_test[:, [feature]]
    xn_train, xn_test = normalize(x_train, x_test)
    w = fit_ls(poly_design(xn_train, degree), t_train)

    
    lo = min(x_train.min(), x_test.min())
    hi = max(x_train.max(), x_test.max())
    grid = np.linspace(lo, hi, 400)[:, None]
    grid_n = normalize(x_train, grid)[1]
    curve = poly_design(grid_n, degree) @ w

    if ax is not None:
        ax.scatter(x_train, t_train, s=18, c="tab:blue", label="train")
        ax.scatter(x_test, t_test, s=10, c="tab:orange", alpha=0.6,
                   label="test")
        ax.plot(grid, curve, "k-", lw=2, label=f"degree {degree} fit")
        ax.set_xlabel(f"{FEATURES[feature]} (feature {feature})")
        ax.set_ylabel("MEDV ($1000s)")
        ax.set_ylim(-20, 80)
        ax.legend()
    return (rmse(poly_design(xn_train, degree), w, t_train),
            rmse(poly_design(xn_test, degree), w, t_test))


LAMBDAS = [0, 0.01, 0.1, 1, 10, 100, 1000]


def polynomial_regression_reg(X_train, t_train, degree=8, feature=2,
                              lambdas=LAMBDAS, k=10, seed=0):
    """10-fold CV for L2-regularized polynomial regression on one feature."""
    x = X_train[:, [feature]]
    n = len(x)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    cv_err = {lam: [] for lam in lambdas}
    for i in range(k):
        val = folds[i]
        trn = np.concatenate([folds[j] for j in range(k) if j != i])
        xn_trn, xn_val = normalize(x[trn], x[val])   
        Phi_trn = poly_design(xn_trn, degree)
        Phi_val = poly_design(xn_val, degree)
        for lam in lambdas:
            w = fit_ridge(Phi_trn, t_train[trn], lam)
            cv_err[lam].append(rmse(Phi_val, w, t_train[val]))
    return {lam: float(np.mean(e)) for lam, e in cv_err.items()}
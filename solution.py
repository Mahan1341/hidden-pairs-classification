import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

threadpool_limits(limits=4)

warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL"
)
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores"
)

random.seed(42)
np.random.seed(42)

START = time.perf_counter()

FEATURES = [f"f{i:04d}" for i in range(512)]

REFERENCE_FILE = Path(
    "submission_D_PCA128_LogPCA128_63_27_10.csv"
)
OUTPUT_FILE = Path("final_submission.csv")

print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)

# ============================================================
# DATA
# ============================================================

for filename in ("train.csv", "validation.csv", "test.csv"):
    if not Path(filename).is_file():
        raise FileNotFoundError(filename)

train = pd.read_csv("train.csv")
validation = pd.read_csv("validation.csv")
test = pd.read_csv("test.csv")

for name, frame in [
    ("train", train),
    ("validation", validation),
    ("test", test)
]:
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing features {missing[:5]}")

assert "target" in train
assert "target" in validation
assert "row_id" in test
assert test["row_id"].is_unique

X = train[FEATURES].to_numpy(dtype=np.float64)
y = train["target"].to_numpy(dtype=np.int64)

X_val = validation[FEATURES].to_numpy(dtype=np.float64)
y_val = validation["target"].to_numpy(dtype=np.int64)

X_test = test[FEATURES].to_numpy(dtype=np.float64)

assert X.shape == (2520, 512)
assert X_val.shape == (560, 512)
assert X_test.shape == (1116, 512)

assert np.array_equal(np.bincount(y, minlength=2), [1260, 1260])
assert np.array_equal(np.bincount(y_val, minlength=2), [280, 280])

assert np.isfinite(X).all()
assert np.isfinite(X_val).all()
assert np.isfinite(X_test).all()

assert X.min() >= 0
assert X_val.min() >= 0
assert X_test.min() >= 0

print("Data:", X.shape, X_val.shape, X_test.shape)

# ============================================================
# D v2 - SUPERVISED POLAR GEOMETRY
# ============================================================

def fit_geometry_model(X_train, y_train, k=32):
    radius = np.linalg.norm(X_train, axis=1)
    U = X_train / np.maximum(radius[:, None], 1e-12)

    lr = np.log1p(radius)
    mu, sigma = lr.mean(), lr.std()
    sigma = max(float(sigma), 1e-12)
    rz = (lr - mu) / sigma

    models = []

    for cls in (0, 1):
        mask = y_train == cls

        U_cls = U[mask]
        rz_cls = rz[mask]

        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = km.fit_predict(U_cls)

        centroids = np.zeros(
            (k, X_train.shape[1]),
            dtype=np.float64
        )
        rhos = np.zeros(k, dtype=np.float64)

        for j in range(k):
            m = labels == j

            if m.any():
                centroids[j] = U_cls[m].mean(axis=0)
                rhos[j] = np.median(rz_cls[m])
            else:
                centroids[j] = km.cluster_centers_[j]
                rhos[j] = np.median(rz_cls)

            centroids[j] /= max(
                np.linalg.norm(centroids[j]), 1e-12
            )

        models.append((centroids, rhos))

    return mu, sigma, models


def transform_geometry(model, X_input):
    mu, sigma, models = model

    radius = np.linalg.norm(X_input, axis=1)
    U = X_input / np.maximum(radius[:, None], 1e-12)

    rz = (np.log1p(radius) - mu) / sigma

    n = len(X_input)
    parts = [rz[:, None]]

    winner_d = []
    winner_a = []
    winner_b = []

    for centroids, rho in models:
        A = 1 - U @ centroids.T
        B = (rz[:, None] - rho[None, :]) ** 2
        D = A + 0.1 * B

        order = np.argsort(D, axis=1)

        nearest = np.take_along_axis(
            D, order[:, :8], axis=1
        )

        win = order[:, 0]
        a_win = A[np.arange(n), win]
        b_win = B[np.arange(n), win]

        parts.extend([
            nearest,
            a_win[:, None],
            b_win[:, None],
            A.min(axis=1, keepdims=True),
            B.min(axis=1, keepdims=True),
            (nearest[:, 1] - nearest[:, 0])[:, None],
        ])

        winner_d.append(nearest[:, 0])
        winner_a.append(a_win)
        winner_b.append(b_win)

    parts.extend([
        (winner_d[0] - winner_d[1])[:, None],
        (winner_a[0] - winner_a[1])[:, None],
        (winner_b[0] - winner_b[1])[:, None],
    ])

    result = np.column_stack(parts)

    assert result.shape == (n, 30)

    return result


def crossfit_geometry_features(
    X_train, y_train, n_splits=5, random_state=137
):
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    features = np.empty(
        (len(X_train), 30),
        dtype=np.float64
    )

    for fit_idx, held_idx in splitter.split(
        X_train, y_train
    ):
        local_model = fit_geometry_model(
            X_train[fit_idx],
            y_train[fit_idx]
        )

        features[held_idx] = transform_geometry(
            local_model,
            X_train[held_idx]
        )

    return features


GEO_KEEP = np.array([
    i for i in range(30) if i not in (13, 26)
])

WEIGHTS = dict(
    raw=0.288,
    rank=0.288,
    pca=0.160,
    direction=0.200,
    geometry=0.064
)

assert len(GEO_KEEP) == 28
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-12


def unit_directions(A):
    return A / np.maximum(
        np.linalg.norm(A, axis=1, keepdims=True),
        1e-12
    )

# ============================================================
# D v2 - FIVE-KERNEL SVM
# ============================================================

def fit_dv2(X_train, y_train):
    t0 = time.perf_counter()

    raw_scaler = StandardScaler()
    Z_train = raw_scaler.fit_transform(X_train)

    rank_transformer = QuantileTransformer(
        n_quantiles=256,
        output_distribution="normal",
        subsample=1_000_000,
        random_state=42
    )

    R_train = rank_transformer.fit_transform(X_train)

    pca = PCA(
        n_components=64,
        whiten=True,
        svd_solver="randomized",
        random_state=42
    )

    P_train = pca.fit_transform(Z_train)

    U_train = unit_directions(X_train)
    gamma_direction = 1 / (512 * U_train.var())

    G_train = crossfit_geometry_features(
        X_train,
        y_train,
        n_splits=5,
        random_state=137
    )

    geometry_model = fit_geometry_model(
        X_train, y_train, k=32
    )

    geometry_scaler = StandardScaler()

    G_train = geometry_scaler.fit_transform(
        G_train
    )[:, GEO_KEEP]

    train_reps = dict(
        raw=Z_train,
        rank=R_train,
        pca=P_train,
        direction=U_train,
        geometry=G_train
    )

    gammas = dict(
        raw=1/512,
        rank=1/512,
        pca=1/64,
        direction=gamma_direction,
        geometry=1/30
    )

    K_train = np.zeros(
        (len(X_train), len(X_train)),
        dtype=np.float64
    )

    for name, rep in train_reps.items():
        K_train += WEIGHTS[name] * rbf_kernel(
            rep,
            gamma=gammas[name]
        )

    classifier = SVC(
        C=2.0,
        kernel="precomputed"
    )

    classifier.fit(K_train, y_train)

    fitted = dict(
        raw_scaler=raw_scaler,
        rank_transformer=rank_transformer,
        pca=pca,
        geometry_model=geometry_model,
        geometry_scaler=geometry_scaler,
        train_reps=train_reps,
        gammas=gammas,
        classifier=classifier
    )

    return fitted, time.perf_counter() - t0


def decision_dv2(fitted, X_new):
    z = fitted["raw_scaler"].transform(X_new)

    reps = dict(
        raw=z,
        rank=fitted["rank_transformer"].transform(X_new),
        pca=fitted["pca"].transform(z),
        direction=unit_directions(X_new),
        geometry=fitted["geometry_scaler"].transform(
            transform_geometry(
                fitted["geometry_model"],
                X_new
            )
        )[:, GEO_KEEP]
    )

    n_train = len(fitted["train_reps"]["raw"])

    kernel = np.zeros(
        (len(X_new), n_train),
        dtype=np.float64
    )

    for name, rep in reps.items():
        kernel += WEIGHTS[name] * rbf_kernel(
            rep,
            fitted["train_reps"][name],
            gamma=fitted["gammas"][name]
        )

    return fitted["classifier"].decision_function(kernel)

# ============================================================
# FIT FULL D v2 ON TRAIN ONLY
# ============================================================

d_model, d_fit_seconds = fit_dv2(X, y)

print(f"D v2 fitted: {d_fit_seconds:.2f}s")
print("D support vectors:", len(d_model["classifier"].support_))

# ============================================================
# FIT ORIGINAL PCA128 ON TRAIN ONLY
# ============================================================

caps = np.quantile(X, 0.995, axis=0)

A = np.minimum(X, caps)
B_val = np.minimum(X_val, caps)

pca_original = PCA(
    n_components=128,
    whiten=False,
    svd_solver="full"
)

P_train = pca_original.fit_transform(A)
P_val = pca_original.transform(B_val)

gamma_original = 2.0 / (
    P_train.shape[1] * P_train.var()
)

svc_original = SVC(
    kernel="rbf",
    C=3.0,
    gamma=gamma_original
)

svc_original.fit(P_train, y)

# ============================================================
# FIT LOG-PCA128 ON TRAIN ONLY
# ============================================================

L_train = np.log1p(A)
L_val = np.log1p(B_val)

pca_log = PCA(
    n_components=128,
    whiten=False,
    svd_solver="full"
)

LP_train = pca_log.fit_transform(L_train)
LP_val = pca_log.transform(L_val)

gamma_log = 2.0 / (
    LP_train.shape[1] * LP_train.var()
)

svc_log = SVC(
    kernel="rbf",
    C=3.0,
    gamma=gamma_log
)

svc_log.fit(LP_train, y)


def rank01(scores):
    return rankdata(scores, method="average") / len(scores)


d_val = decision_dv2(d_model, X_val)
p_val = svc_original.decision_function(P_val)
log_val = svc_log.decision_function(LP_val)

rd_val = rank01(d_val)
rp_val = rank01(p_val)
rl_val = rank01(log_val)

baseline_val = (
    0.70 * rd_val
    + 0.30 * rp_val
)

candidate_val = (
    0.63 * rd_val
    + 0.27 * rp_val
    + 0.10 * rl_val
)

reference_aucs = {
    "D v2": (
        roc_auc_score(y_val, d_val),
        0.95025510
    ),
    "PCA128": (
        roc_auc_score(y_val, p_val),
        0.95061224
    ),
    "Log-PCA128": (
        roc_auc_score(y_val, log_val),
        0.95237245
    ),
    "Original 70/30": (
        roc_auc_score(y_val, baseline_val),
        0.95397321
    ),
    "Final 63/27/10": (
        roc_auc_score(y_val, candidate_val),
        0.95454082
    )
}

print("\nVALIDATION REPRODUCTION")
print("=" * 65)

for name, (actual, expected) in reference_aucs.items():
    print(
        f"{name:20s} "
        f"actual={actual:.8f} "
        f"expected={expected:.8f}"
    )

    if abs(actual - expected) > 1e-6:
        raise RuntimeError(
            f"{name}: validation reproduction FAILED"
        )

print("Validation reproduction: PASS")

# ============================================================
# TEST INFERENCE ONLY
# ============================================================

t_inference = time.perf_counter()

B_test = np.minimum(X_test, caps)

P_test = pca_original.transform(B_test)
LP_test = pca_log.transform(np.log1p(B_test))

d_test = decision_dv2(d_model, X_test)
p_test = svc_original.decision_function(P_test)
log_test = svc_log.decision_function(LP_test)

rd_test = rank01(d_test)
rp_test = rank01(p_test)
rl_test = rank01(log_test)

test_pred = (
    0.63 * rd_test
    + 0.27 * rp_test
    + 0.10 * rl_test
)

inference_seconds = time.perf_counter() - t_inference

assert test_pred.shape == (1116,)
assert np.isfinite(test_pred).all()
assert ((test_pred >= 0) & (test_pred <= 1)).all()

submission = pd.DataFrame({
    "row_id": test["row_id"],
    "target": test_pred
})

assert submission["row_id"].equals(test["row_id"])

# ============================================================
# EXACT COMPARISON WITH ORIGINAL KAGGLE SUBMISSION
# ============================================================

reference_verified = False

if REFERENCE_FILE.exists():
    reference = pd.read_csv(REFERENCE_FILE)

    assert list(reference.columns) == ["row_id", "target"]
    assert reference["row_id"].equals(submission["row_id"])

    original_pred = reference["target"].to_numpy(
        dtype=np.float64
    )

    assert np.isfinite(original_pred).all()

    max_diff = float(
        np.max(np.abs(original_pred - test_pred))
    )

    print("\nORIGINAL SUBMISSION COMPARISON")
    print("Reference:", REFERENCE_FILE)
    print(f"Maximum absolute difference: {max_diff:.12g}")

    if max_diff > 1e-6:
        raise RuntimeError(
            "FAILED: predictions differ from Kaggle submission"
        )

    reference_verified = True
    print("Exact submission reproduction: PASS")

else:
    print(
        "\nWARNING: original Kaggle CSV not found. "
        "Exact submission reproduction is NOT verified."
    )

# ============================================================
# SAVE AND VERIFY FINAL CSV
# ============================================================

submission.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.17g"
)

saved = pd.read_csv(OUTPUT_FILE)

assert list(saved.columns) == ["row_id", "target"]
assert saved["row_id"].equals(test["row_id"])

np.testing.assert_allclose(
    saved["target"].to_numpy(dtype=np.float64),
    test_pred,
    rtol=0,
    atol=1e-12
)

elapsed = time.perf_counter() - START

print("\nFINAL CHECK")
print("=" * 65)
print("Output:", OUTPUT_FILE.resolve())
print("Rows:", len(saved))
print("Validation: PASS")
print("Original CSV verified:", reference_verified)
print(f"Test inference: {inference_seconds:.2f}s")
print(f"Full runtime: {elapsed:.2f}s")
print("CPU 15-minute target:", "PASS" if elapsed < 900 else "EXCEEDED")
print("Inference 2-minute target:", "PASS" if inference_seconds < 120 else "EXCEEDED")
print("No cache, pretrained weights, external data or test fitting.")

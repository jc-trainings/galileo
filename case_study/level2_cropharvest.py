"""Level 2 — map maize: frozen Galileo + a small classifier trained on labels.

Question: where is the maize?
Galileo's weights are never changed here. Only a small classifier is trained, on top of the
128 numbers (embeddings) that Galileo produces for each pixel.

What the script does, step by step:
  STEP 1. Load the CropHarvest Kenya labels:
          3,940 training points (264 maize) and 898 test pixels in two Busia areas.
  STEP 2. Get the embeddings of the labelled points with frozen Galileo (saved for reuse).
  STEP 3. Label efficiency: train logistic regression and k-NN with 20 to 3,940 labels and
          test each on the Busia areas            -> results_summary.csv, label_efficiency.png
  STEP 4. Tune four classifiers by cross-validation on the training points only, in ~5 km
          spatial blocks (so neighbouring fields cannot leak between training and checking).
  STEP 5. Give each classifier its own maize/not-maize cut-off, chosen on training data.
  STEP 6. Test every classifier on the Busia areas, overall and per area, next to two
          baselines: a random forest on the raw bands (no Galileo) and "call every pixel
          maize"                                   -> model_comparison.csv
  STEP 7. Save the mapping classifier and draw the test areas
                                                   -> maize_classifier.joblib, test_regions.png
  STEP 8. (with --tif) Map your area with the chosen classifier (k-NN by default) and check
          the map against WorldCereal's maize map -> maize_probability.tif, maize_class.tif


Example:
    python case_study/level2_cropharvest.py
    python case_study/level2_cropharvest.py --tif "data/busia/<exported file>.tif"
"""

import argparse
import re
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    REPO_ROOT,
    galileo_normalizer,
    interior_mask,
    load_model,
    pixel_latlons,
    run_encoder,
    tif_profile,
    worldcereal_check,
    write_geotiff,
)
from cropharvest_download import ensure_cropharvest  
from src.data.dataset import SPACE_BANDS, SPACE_TIME_BANDS, TIME_BANDS, Dataset 
from src.eval.cropharvest.bands import BANDS as CH_BANDS 
from src.eval.cropharvest.cropharvest_eval import ( 
    BinaryCropHarvestEval,
    CropHarvestEvalBase,
)

SEED = 42
CH_TIMESTEPS = 12  # CropHarvest arrays: 12 monthly steps (Feb 2019 - Jan 2020 for Kenya)


# ---------------------------------------------------------------------------
# Data (used in STEP 1)
# ---------------------------------------------------------------------------
def load_kenya(normalizer):
    """CropHarvest Kenya maize: training points and the held-out Busia test pixels.

    Downloads to data/cropharvest_data on first use. Returns the evaluator (used as the
    preprocessing "converter"), (x, latlons, y) for training and
    (x, latlons, y, test_area_name) for testing. x is [N, 12, len(CH_BANDS)].
    """
    ensure_cropharvest()  # resumable, size-checked download (the repo's own one isn't)
    ev = BinaryCropHarvestEval(country="Kenya", normalizer=normalizer, eval_mode="test")
    x_train, ll_train, y_train = ev.dataset.as_array()
    print(f"Training points: {len(y_train)} ({int(y_train.sum())} maize, "
          f"{int((y_train == 0).sum())} not maize)")

    x_test, ll_test, y_test, region = [], [], [], []
    for test_id, inst in ev.dataset.test_data(max_size=10000):
        keep = inst.y != -1  # -1 = unlabelled pixel
        x_test.append(inst.x[keep])
        ll_test.append(np.stack([inst.lats[keep], inst.lons[keep]], -1))
        y_test.append(inst.y[keep])
        region += [re.sub(r"^\d+_", "", test_id)] * int(keep.sum())  # drop chunk prefix
    x_test, ll_test = np.concatenate(x_test), np.concatenate(ll_test)
    y_test, region = np.concatenate(y_test), np.array(region)
    print(f"Test pixels: {len(y_test)} ({int(y_test.sum())} maize) in "
          f"{len(set(region))} test areas: {sorted(set(region))}")
    return ev, (x_train, ll_train, y_train), (x_test, ll_test, y_test, region)


# ---------------------------------------------------------------------------
# Embedding (used in STEP 2 and STEP 8)
#
# Same idea as Level 1 (common.embed_tif_full_inputs), with one difference: only the inputs
# CropHarvest has are used (Sentinel-1, Sentinel-2, NDVI, temperature, elevation, location).
# The rest are masked. Map pixels must be described the same way as the labelled points,
# or the classifier would be comparing different things.
# ---------------------------------------------------------------------------
def embed_cropharvest_array(model, converter: CropHarvestEvalBase, array, latlons,
                            batch_size=512, desc="Embedding"):
    """array: [N, T, len(CH_BANDS)] raw CropHarvest-format values -> [N, D] embeddings.

    `converter.cropharvest_array_to_normalized_galileo` is the repo's own function: it
    puts each band in the right slot, masks the groups CropHarvest doesn't have
    (Dynamic World, WorldCereal, TerraClimate, VIIRS, LandScan), adds lat/lon and
    normalizes. Using it for both training points and map pixels keeps them identical.
    """
    out = []
    # Go through the points in batches: prepare them the way Galileo expects, run Galileo,
    # keep the 128 numbers per point.
    for i in tqdm(range(0, len(array), batch_size), desc=desc):
        mo = converter.cropharvest_array_to_normalized_galileo(
            array[i:i + batch_size], latlons[i:i + batch_size],
            start_month=converter.start_month, timesteps=CH_TIMESTEPS,
        )
        out.append(run_encoder(model, mo))
    return np.concatenate(out)


def tif_to_cropharvest_array(tif_path: Path) -> np.ndarray:
    """Rearrange a Galileo-format GeoTIFF into CropHarvest's per-pixel layout.

    Returns [H*W, 12, len(CH_BANDS)] in raw units, so map pixels go through exactly
    the same preprocessing as the CropHarvest training points.
    """
    # Read the export into Galileo's input groups (same as Level 1, step A), unscaled
    dso = Dataset._tif_to_array(tif_path)
    st, sp, tx = dso.space_time_x, dso.space_x, dso.time_x  # [H,W,T,13], [H,W,16], [T,6]
    h, w, t, _ = st.shape
    if t < CH_TIMESTEPS:
        raise ValueError(f"{tif_path.name} has {t} timesteps; need at least {CH_TIMESTEPS}. "
                         "Export a full Jan-Dec year.")
    if t > CH_TIMESTEPS:
        print(f"Using the first {CH_TIMESTEPS} of {t} timesteps")
    st, tx = st[:, :, :CH_TIMESTEPS], tx[:CH_TIMESTEPS]

    # Copy each CropHarvest band from the matching place in the export, pixel by pixel
    arr = np.zeros((h * w, CH_TIMESTEPS, len(CH_BANDS)), dtype=np.float32)
    for j, band in enumerate(CH_BANDS):
        if band in SPACE_TIME_BANDS:
            arr[:, :, j] = st[..., SPACE_TIME_BANDS.index(band)].reshape(h * w, CH_TIMESTEPS)
        elif band in TIME_BANDS:
            arr[:, :, j] = tx[:, TIME_BANDS.index(band)][None, :]
        elif band in SPACE_BANDS:
            arr[:, :, j] = sp[..., SPACE_BANDS.index(band)].reshape(h * w, 1)
        # bands Galileo doesn't use (e.g. B9) stay 0 and are dropped by the converter
    return arr


# ---------------------------------------------------------------------------
# Classifiers for the label-efficiency test (used in STEP 3)
# ---------------------------------------------------------------------------
def make_models(n_train: int) -> dict:
    # StandardScaler centres and scales each of the 128 embedding numbers using the
    # training points only; this matters for both k-NN distances and regularization.
    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
        "kNN-5": make_pipeline(StandardScaler(),
                               KNeighborsClassifier(n_neighbors=5, metric="cosine")),
    }
    if n_train > 20:
        models["kNN-20"] = make_pipeline(StandardScaler(),
                                         KNeighborsClassifier(n_neighbors=20, metric="cosine"))
    return models


def balanced_subset(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of n training points, half maize / half not-maize where possible."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k_pos = min(len(pos), n // 2)
    k_neg = min(len(neg), n - k_pos)
    return np.concatenate([rng.choice(pos, k_pos, replace=False),
                           rng.choice(neg, k_neg, replace=False)])


def score(model, e_test, y_test) -> dict:
    prob = model.predict_proba(e_test)[:, 1]
    return {"f1": f1_score(y_test, prob > 0.5), "auc": roc_auc_score(y_test, prob)}



# ---------------------------------------------------------------------------
# Choosing and testing classifiers (used in STEPS 4-6)
# ---------------------------------------------------------------------------
BLOCK_DEG = 0.05  # ~5 km spatial blocks: points in the same block stay in the same CV fold


def spatial_blocks(latlons: np.ndarray) -> np.ndarray:
    """Group points into ~5 km blocks so cross-validation tests on unseen areas."""
    cells = np.floor(latlons / BLOCK_DEG).astype(int)
    _, groups = np.unique(cells, axis=0, return_inverse=True)
    return groups.ravel()


def make_cv(y: np.ndarray, groups: np.ndarray):
    """Spatially grouped 5-fold CV; falls back to plain stratified CV if the blocks are
    too few or too uneven to give every fold both classes."""
    from sklearn.model_selection import StratifiedKFold
    if len(np.unique(groups)) >= 10:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        folds = list(cv.split(np.zeros(len(y)), y, groups))
        if all(len(np.unique(y[te])) == 2 and len(np.unique(y[tr])) == 2 for tr, te in folds):
            return folds, f"spatial blocks of ~5 km ({len(np.unique(groups))} blocks)"
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                 .split(np.zeros(len(y)), y))
    return folds, "random stratified folds (too few spatial blocks for a spatial split)"


def candidates() -> dict:
    """Classifiers to try on the embeddings, each with a small grid of settings."""
    def scale():
        return ("scale", StandardScaler())
    return {
        "Logistic regression": (
            Pipeline([scale(), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000,
                                                        random_state=SEED))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "k-NN": (
            Pipeline([scale(), ("clf", KNeighborsClassifier(metric="cosine"))]),
            {"clf__n_neighbors": [10, 20, 40, 80], "clf__weights": ["uniform", "distance"]}),
        "Random forest": (
            Pipeline([("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                     class_weight="balanced_subsample",
                                                     n_jobs=-1, random_state=SEED))]),
            {"clf__max_features": ["sqrt", 0.3]}),
        "Gradient boosting": (
            Pipeline([("clf", HistGradientBoostingClassifier(class_weight="balanced",
                                                             random_state=SEED))]),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31]}),
    }


def youden_threshold(y, prob) -> float:
    """Cut-off that maximises (true-positive rate - false-positive rate).

    """
    fpr, tpr, thr = roc_curve(y, prob)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0.01, 0.99))


def test_scores(y, prob, threshold) -> dict:
    return {"test_auc": roc_auc_score(y, prob),
            "test_f1@0.5": f1_score(y, prob > 0.5),
            "test_f1@cutoff": f1_score(y, prob > threshold),
            "test_bal_acc@cutoff": balanced_accuracy_score(y, prob > threshold)}


def raw_features(x: np.ndarray) -> np.ndarray:
    """Traditional baseline features: the raw monthly bands, flattened (no foundation model)."""
    return x.reshape(len(x), -1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=REPO_ROOT / "data/models/nano")
    p.add_argument("--sizes", type=int, nargs="+", default=[20, 32, 64, 128, 256, 0],
                   help="Training-set sizes for the label-efficiency curve (0 = all)")
    p.add_argument("--repeats", type=int, default=3, help="Random draws per size")
    p.add_argument("--tif", type=Path, default=None, help="Optional area to map")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/level2")
    p.add_argument("--map_model", default="k-NN",
                   choices=["k-NN", "Logistic regression", "Random forest", "Gradient boosting",
                            "cv-best"],
                   help="Classifier used for the maps. Default k-NN: best on both Busia test "
                        "areas in the reference run. cv-best = highest cross-validated AUC")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    model = load_model(a.model)  # pretrained Galileo from data/models/nano (never changed)
    normalizer = galileo_normalizer()

    # --------------------------------------------------------------------------------
    # STEP 1. Load the CropHarvest Kenya labels
    # --------------------------------------------------------------------------------
    # Downloads to data/cropharvest_data on the first run. x = 12 months of satellite values
    # per point, ll = latitude/longitude, y = 1 for maize, 0 for not maize.
    ev, (x_train, ll_train, y_train), (x_test, ll_test, y_test, region) = load_kenya(normalizer)

    # --------------------------------------------------------------------------------
    # STEP 2. Get the embeddings of the labelled points (frozen Galileo)
    # --------------------------------------------------------------------------------
    # Each point becomes 128 numbers. Saved to a .npz file per model, so reruns are quick.
    cache = a.out / f"cropharvest_kenya_embeddings_{a.model.name}.npz"  # one per model
    if cache.exists():
        z = np.load(cache)
        e_train, e_test = z["e_train"], z["e_test"]
    else:
        e_train = embed_cropharvest_array(model, ev, x_train, ll_train, desc="Train points")
        e_test = embed_cropharvest_array(model, ev, x_test, ll_test, desc="Test pixels")
        np.savez(cache, e_train=e_train, e_test=e_test, y_train=y_train, y_test=y_test)
    print(f"Embedding size: {e_train.shape[1]}")

    # --------------------------------------------------------------------------------
    # STEP 3. Label efficiency: how many labels do we need?
    # --------------------------------------------------------------------------------
    # For each number of labels (20, 32, ... all), pick a random half-maize/half-not set,
    # train logistic regression and k-NN on the embeddings, and test on the Busia areas.
    # Small sizes are repeated 3 times with different random draws and averaged.
    rng = np.random.default_rng(SEED)
    rows = []
    for n in a.sizes:
        full = n == 0 or n >= len(y_train)
        for rep in range(1 if full else a.repeats):
            idx = np.arange(len(y_train)) if full else balanced_subset(y_train, n, rng)
            for name, clf in make_models(len(idx)).items():
                clf.fit(e_train[idx], y_train[idx])
                rows.append({"n_labels": len(idx), "repeat": rep, "model": name,
                             **score(clf, e_test, y_test)})
    res = pd.DataFrame(rows)
    res.to_csv(a.out / "results_all_runs.csv", index=False)
    summary = res.groupby(["n_labels", "model"])[["f1", "auc"]].mean().round(3).unstack("model")
    summary.to_csv(a.out / "results_summary.csv")
    print("\nTest-set scores (mean over repeats):\n", summary.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax in zip(["f1", "auc"], axes):
        for name, g in res.groupby("model"):
            m = g.groupby("n_labels")[metric].mean()
            ax.plot(m.index, m.values, marker="o", label=name)
        ax.set_xscale("log")
        ax.set_xlabel("Number of training labels")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"CropHarvest Kenya maize: {metric.upper()} vs labels")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(a.out / "label_efficiency.png", dpi=150)

    # --------------------------------------------------------------------------------
    # STEP 4. Tune four classifiers by spatial cross-validation (training points only)
    # --------------------------------------------------------------------------------
    # Cross-validation: split the training points into 5 groups, train on 4 and check on the
    # 5th, five times over. Points are grouped in ~5 km blocks so a field and its neighbour
    # never sit on both sides. Each classifier tries a few settings; the best one is kept.
    # The Busia test areas are NOT used here.
    cv, cv_kind = make_cv(y_train, spatial_blocks(ll_train))
    print(f"\nChoosing a classifier by 5-fold cross-validation on the training points "
          f"({cv_kind})...")
    rows, fitted = [], {}
    for name, (pipe, grid) in candidates().items():
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        gs.fit(e_train, y_train)
        fitted[name] = gs
        settings = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in gs.best_params_.items())
        rows.append({"model": f"Galileo + {name}", "settings": settings, "cv_auc": gs.best_score_})
        print(f"  {name:20s} CV AUC {gs.best_score_:.3f}  ({settings})")
    # In the reference run all four score ~0.96 here: most training points are easy
    # "non-crop" points, so cross-validation cannot separate the classifiers. The map uses
    # k-NN by default (best on both test areas and closest to WorldCereal); --map_model
    # cv-best uses the cross-validation choice instead.
    cv_best = max(fitted, key=lambda n: fitted[n].best_score_)
    spread = max(g.best_score_ for g in fitted.values()) - min(g.best_score_ for g in fitted.values())
    if spread < 0.01:
        print(f"  The four CV scores are within {spread:.3f} of each other: the training points "
              "cannot tell these classifiers apart.")
    best_name = cv_best if a.map_model == "cv-best" else a.map_model
    best = fitted[best_name].best_estimator_
    print(f"Classifier used for the maps: {best_name}"
          + (" (highest CV AUC)" if a.map_model == "cv-best" else " (set with --map_model)"))

    # --------------------------------------------------------------------------------
    # STEP 5. A maize/not-maize cut-off for each classifier (training data only)
    # --------------------------------------------------------------------------------
    # A classifier gives a probability; the cut-off turns it into yes/no. Instead of assuming
    # 0.5, we take each classifier's cross-validation predictions on the training points and
    # pick the cut-off that best balances finding maize against false alarms (Youden's J).
    cutoffs = {}
    for name in fitted:
        oof = cross_val_predict(fitted[name].best_estimator_, e_train, y_train, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        cutoffs[name] = youden_threshold(y_train, oof)
    cutoff = cutoffs[best_name]
    print(f"Maize cut-off for {best_name}, chosen on training data: {cutoff:.2f}")

    # --------------------------------------------------------------------------------
    # STEP 6. Test every classifier on the Busia areas, next to two baselines
    # --------------------------------------------------------------------------------
    # Scores: AUC (how well maize is ranked above not-maize; 0.5 = guessing, 1 = perfect),
    # F1 and balanced accuracy at the cut-off. AUC is also given per test area: the two
    # areas differ (54% vs 79% maize), and a model whose probabilities shift between areas
    # scores worse on both areas together.
    regions = sorted(set(region))

    def area_aucs(prob):
        return {f"test_auc_area{i + 1}": roc_auc_score(y_test[region == rg], prob[region == rg])
                for i, rg in enumerate(regions)}

    for row, name in zip(rows, fitted):
        prob = fitted[name].predict_proba(e_test)[:, 1]
        row.update(test_scores(y_test, prob, cutoffs[name]))
        row.update(area_aucs(prob))
        row["cutoff"] = cutoffs[name]
        row["used_for_map"] = name == best_name
    # Baseline 1: the traditional way, a random forest on the raw monthly bands (no Galileo)
    trad = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    trad_cv = cross_val_predict(trad, raw_features(x_train), y_train, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
    trad.fit(raw_features(x_train), y_train)
    prob_trad = trad.predict_proba(raw_features(x_test))[:, 1]
    rows.append({"model": "Traditional: random forest on raw bands (no Galileo)", "settings": "",
                 "cv_auc": roc_auc_score(y_train, trad_cv), "used_for_map": False,
                 **test_scores(y_test, prob_trad, youden_threshold(y_train, trad_cv)),
                 **area_aucs(prob_trad), "cutoff": youden_threshold(y_train, trad_cv)})
    # Baseline 2: no skill at all, call every pixel maize (shows what "good" has to beat)
    rows.append({"model": "No skill: call every pixel maize", "settings": "", "cv_auc": 0.5,
                 "used_for_map": False, "test_auc": 0.5,
                 **{f"test_auc_area{i + 1}": 0.5 for i in range(len(regions))},
                 "test_f1@0.5": f1_score(y_test, np.ones_like(y_test)),
                 "test_f1@cutoff": f1_score(y_test, np.ones_like(y_test)),
                 "test_bal_acc@cutoff": 0.5})
    comp = pd.DataFrame(rows).round(3)
    comp.to_csv(a.out / "model_comparison.csv", index=False)
    print("\nComparison on the held-out Busia test areas:")
    area_cols = [f"test_auc_area{i + 1}" for i in range(len(regions))]
    print(comp[["model", "cv_auc", "test_auc", *area_cols, "cutoff", "test_f1@cutoff",
                "test_bal_acc@cutoff", "used_for_map"]].to_string(index=False))

    # --------------------------------------------------------------------------------
    # STEP 7. Save the mapping classifier and draw the test areas
    # --------------------------------------------------------------------------------
    # test_regions.png: for each test area, the true labels, the predicted probability and
    # the yes/no answer at the cut-off, side by side.
    joblib.dump({"model": best, "cutoff": cutoff, "name": best_name},
                a.out / "maize_classifier.joblib")
    prob_test = best.predict_proba(e_test)[:, 1]
    fig, axes = plt.subplots(len(regions), 3, figsize=(14, 4.5 * len(regions)), squeeze=False)
    for i, rg in enumerate(regions):
        m = region == rg
        panels = [(y_test[m], "Ground truth (1 = maize)"),
                  (prob_test[m], f"Predicted maize probability ({best_name})"),
                  ((prob_test[m] > cutoff).astype(float), f"Predicted maize (cut-off {cutoff:.2f})")]
        for j, (vals, title) in enumerate(panels):
            sc = axes[i, j].scatter(ll_test[m, 1], ll_test[m, 0], c=vals, cmap="RdYlGn",
                                    vmin=0, vmax=1, s=6, marker="s")
            axes[i, j].set_title(f"{rg}\n{title}", fontsize=9)
            axes[i, j].set_aspect("equal")
            fig.colorbar(sc, ax=axes[i, j], fraction=0.046)
    fig.tight_layout()
    fig.savefig(a.out / "test_regions.png", dpi=150)

    # --------------------------------------------------------------------------------
    # STEP 8. (with --tif) Map your area and check the map against WorldCereal
    # --------------------------------------------------------------------------------
    if a.tif is not None:
        print(f"\nMapping {a.tif.name} with {best_name}")
        # 8a. Describe every pixel the same way as the labelled points, then embed it
        arr = tif_to_cropharvest_array(a.tif)
        latlons = pixel_latlons(a.tif)
        e_map = embed_cropharvest_array(model, ev, arr, latlons, desc="Map pixels")
        # 8b. The classifier gives each pixel a maize probability (0-1)
        prof = tif_profile(a.tif)
        prob = best.predict_proba(e_map)[:, 1].reshape(prof["height"], prof["width"])
        prob[~interior_mask(*prob.shape)] = np.nan  # the export's bad outer border
        # 8c. Yes/no map at the cut-off from step 5 (255 = no-data border)
        cls = np.where(np.isfinite(prob), prob > cutoff, 255).astype(np.uint8)
        write_geotiff(a.out / "maize_probability.tif", prob.astype(np.float32), a.tif,
                      nodata=np.nan)
        write_geotiff(a.out / "maize_class.tif", cls, a.tif, nodata=255)
        # 8d. Picture of both maps
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im = axes[0].imshow(prob, cmap="RdYlGn", vmin=0, vmax=1)
        fig.colorbar(im, ax=axes[0], label="P(maize)")
        axes[0].set_title(f"Maize probability ({best_name})")
        axes[1].imshow(np.ma.masked_equal(cls, 255), cmap="RdYlGn", vmin=0, vmax=1)
        axes[1].set_title(f"Maize (probability > {cutoff:.2f})")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(a.out / "maize_probability.png", dpi=150, bbox_inches="tight")
        print(f"  {(prob[np.isfinite(prob)] > cutoff).mean() * 100:.1f}% of pixels predicted "
              f"maize at cut-off {cutoff:.2f}")
        # 8e. Compare with WorldCereal's maize map (independent, not used for training)
        worldcereal_check(a.tif, prob, cutoff)

    print(f"\nOutputs written to {a.out}")


if __name__ == "__main__":
    main()

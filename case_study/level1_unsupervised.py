"""Level 1 — explore the landscape: frozen Galileo, no labels, no training.

Question: how is this area organised, and where else looks like a place I choose?

What the script does, step by step (the same numbers are used in the comments below):
  STEP 1. Get the embeddings: run every pixel of your export through the pretrained Galileo
          model (see common.embed_tif_full_inputs). Each pixel becomes 128 numbers.
          Saved to embeddings_<model>_<id>.npy, so a rerun takes seconds.
  STEP 2. Leave out the outer 50 m border of the export (bad values there).
  STEP 3. Prepare the numbers: standardise them, then keep the main PCA patterns
          (95% of the variation) and drop the noise.
  STEP 4. Guide for how many zones: silhouette score for 3-10 zones.
  STEP 5. Make zones with k-means (8 by default), lightly smoothed   -> clusters.tif
  STEP 6. Describe each zone with Dynamic World land cover, and compare with the same
          k-means on the raw bands (no Galileo)                       -> printed
  STEP 7. Colour view: the first 3 PCA patterns as red, green, blue   -> pca_rgb.tif
  STEP 8. Similarity search: how much each pixel resembles a point
          you choose with --ref_lat / --ref_lon                        -> similarity.tif
  STEP 9. Quick-look picture of steps 5, 7 and 8                      -> overview.png

Galileo's weights are never changed here: it is used only to turn pixels into numbers.

Example:
    python case_study/level1_unsupervised.py \
        --tif "data/busia/<exported file>.tif" --clusters 8 \
        --ref_lat 0.4747 --ref_lon 34.2306
"""

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    EDGE_PX,
    REPO_ROOT,
    embed_tif_full_inputs,
    interior_mask,
    load_model,
    nearest_pixel,
    write_geotiff,
)

MAX_COMPONENTS = 32       # cap on PCA components used for clustering
SILHOUETTE_SAMPLE = 5000  # pixels used to score each number of zones


def majority_filter(labels: np.ndarray, k: int) -> np.ndarray:
    """3x3 majority vote: removes isolated 'salt-and-pepper' pixels from the zone map."""
    counts = np.stack([uniform_filter((labels == i).astype(np.float32), size=3, mode="nearest")
                       for i in range(k)])
    return counts.argmax(0).astype(labels.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tif", type=Path, required=True)
    p.add_argument("--model", type=Path, default=REPO_ROOT / "data/models/nano")
    p.add_argument("--clusters", type=int, default=8,
                   help="Number of zones; 0 = choose by silhouette score")
    p.add_argument("--smooth", type=int, default=1, help="1 = 3x3 majority smoothing of zones")
    p.add_argument("--ref_lat", type=float, default=None)
    p.add_argument("--ref_lon", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/level1")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------------
    # STEP 1. Get the embeddings: 128 numbers for every pixel
    # --------------------------------------------------------------------------------
    # If this export was embedded before with this model, reuse the saved file. The name
    # depends on the model and the input file, so changing either makes new embeddings.
    key = hashlib.md5(f"{a.model.resolve()}|{a.tif.resolve()}".encode()).hexdigest()[:8]
    emb_path = a.out / f"embeddings_{a.model.name}_{key}.npy"
    if emb_path.exists():
        print(f"Re-using {emb_path}")
        emb = np.load(emb_path)
    else:
        # Otherwise make them: load pretrained Galileo and run every pixel through it.
        # The detailed steps (A-F) are in common.py, function embed_tif_full_inputs.
        emb = embed_tif_full_inputs(load_model(a.model), a.tif, a.batch_size)
        np.save(emb_path, emb)
    h, w, d = emb.shape  # rows, columns, 128

    # --------------------------------------------------------------------------------
    # STEP 2. Leave out the export's bad outer border (5 pixels = 50 m)
    # --------------------------------------------------------------------------------
    # `inside` is True for pixels we keep; everything below works on those pixels only.
    inside = interior_mask(h, w).ravel()
    flat = emb.reshape(-1, d)[inside]
    print(f"Embeddings: {h} x {w} pixels, {d} numbers each "
          f"(outer {EDGE_PX}-pixel border left out: bad values in the export)")

    # --------------------------------------------------------------------------------
    # STEP 3. Prepare the numbers: standardise, then keep the main PCA patterns
    # --------------------------------------------------------------------------------
    # Standardise: put each of the 128 numbers on the same scale (mean 0, spread 1).
    # PCA: find the main patterns in the 128 numbers and keep enough of them to carry 95% of
    # the variation (23 in Busia). The rest is mostly noise.
    # (We tried "whitening" too, which gives every pattern equal weight; it made the zones
    # match land cover less well, because it boosts the weak, noisy patterns.)
    z = StandardScaler().fit_transform(flat)
    pca = PCA(n_components=0.95, random_state=0).fit(z)
    if pca.n_components_ > MAX_COMPONENTS:
        pca = PCA(n_components=MAX_COMPONENTS, random_state=0).fit(z)
    feats = pca.transform(z)
    print(f"Clustering space: {feats.shape[1]} components "
          f"({pca.explained_variance_ratio_.sum() * 100:.0f}% of the variation)")

    # --------------------------------------------------------------------------------
    # STEP 4. How many zones? Silhouette score as a guide
    # --------------------------------------------------------------------------------
    # The silhouette score (-1 to 1) says how cleanly pixels fall into separate zones.
    # It is computed on a sample of 5,000 pixels for 3 to 10 zones, only as a guide:
    # landscapes change gradually, so there is rarely one "right" number.
    rng = np.random.default_rng(0)
    sample = feats[rng.choice(len(feats), min(SILHOUETTE_SAMPLE, len(feats)), replace=False)]
    scores = {}
    for k in range(3, 11):
        lab = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(sample)
        scores[k] = silhouette_score(sample, lab)
    best_k = max(scores, key=scores.get)
    print("Silhouette score by number of zones (guide only; higher = better separated):")
    for k, sc in scores.items():
        print(f"  k={k:2d}: {sc:.3f}{'  <- highest' if k == best_k else ''}")
    if best_k == max(scores):
        print("  Scores keep rising with k, so there is no clear best number of zones:")
        print("  choose k by comparing the zone map with satellite imagery.")
    k = best_k if a.clusters == 0 else a.clusters

    # --------------------------------------------------------------------------------
    # STEP 5. Make zones with k-means (no labels)
    # --------------------------------------------------------------------------------
    # k-means puts pixels with similar numbers in the same zone. It is never told what the
    # zones are: you name them afterwards by comparing with imagery (and step 6).
    # Zones are numbered from largest (0) to smallest.
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(feats)
    order = np.argsort(-np.bincount(km.labels_, minlength=k))
    relabel = np.empty(k, dtype=np.uint8)
    relabel[order] = np.arange(k)
    NODATA = 255
    inside2d = inside.reshape(h, w)
    raw = np.full(h * w, NODATA, dtype=np.uint8)
    raw[inside] = relabel[km.labels_]
    raw = raw.reshape(h, w)
    # Light smoothing: each pixel takes the most common zone among its 3x3 neighbours,
    # which removes single-pixel speckle. The unsmoothed zones are saved too.
    clusters = majority_filter(np.where(inside2d, raw, 0), k) if a.smooth else raw.copy()
    clusters[~inside2d] = NODATA
    write_geotiff(a.out / "clusters.tif", clusters, a.tif, nodata=NODATA)
    write_geotiff(a.out / "clusters_unsmoothed.tif", raw, a.tif, nodata=NODATA)
    sizes = np.bincount(clusters[inside2d], minlength=k) / inside.sum() * 100
    print(f"Zones (k={k}{', smoothed 3x3' if a.smooth else ''}):")
    for i, pct in enumerate(sizes):
        print(f"  zone {i}: {pct:5.1f}% of area")

    # --------------------------------------------------------------------------------
    # STEP 6. Describe the zones with Dynamic World, and compare with raw bands
    # --------------------------------------------------------------------------------
    # Dynamic World is a separate land-cover map included in the export. For each zone we
    # print its land-cover mix ("zone 6: trees 96%") to help you name the zones.
    # Then a fair test of what Galileo adds: run the same k-means on the raw 233 bands (no
    # Galileo) and measure how well each set of zones matches Dynamic World (normalised
    # mutual information, 0 = no match, 1 = perfect). Galileo also gets the Dynamic World
    # bands as inputs, so the raw-band version includes them too.
    import rasterio
    from sklearn.metrics import normalized_mutual_info_score
    with rasterio.open(a.tif) as src:
        names = list(src.descriptions)
        dw_idx = [i for i, n in enumerate(names) if n and n.startswith("DW_")]
        if dw_idx:
            dw_names = [names[i][3:].replace("_", " ") for i in dw_idx]
            dw = np.stack([src.read(i + 1) for i in dw_idx]).reshape(len(dw_idx), -1)
            dw = np.nan_to_num(dw).argmax(0)
            print("\nWhat each zone contains (Dynamic World land cover, share of zone):")
            dw = dw[inside]
            flat_zones = clusters.ravel()[inside]
            for i in range(k):
                m = flat_zones == i
                mix = np.bincount(dw[m], minlength=len(dw_names)) / max(m.sum(), 1)
                top = np.argsort(-mix)[:3]
                print(f"  zone {i}: " + ", ".join(f"{dw_names[j]} {mix[j] * 100:.0f}%"
                                                  for j in top if mix[j] >= 0.05))
            raw_bands = StandardScaler().fit_transform(
                np.nan_to_num(src.read().astype(np.float32).reshape(src.count, -1).T[inside]))
            raw_zones = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(raw_bands)
            print("Agreement of zones with Dynamic World (normalised mutual information, 0-1):")
            print(f"  Galileo embeddings: {normalized_mutual_info_score(dw, raw.ravel()[inside]):.2f}")
            print(f"  Raw bands, no Galileo: {normalized_mutual_info_score(dw, raw_zones):.2f}")

    # --------------------------------------------------------------------------------
    # STEP 7. Colour view: the first 3 PCA patterns as red, green, blue
    # --------------------------------------------------------------------------------
    # Pixels with similar colours have similar embeddings. Values are stretched between the
    # 2nd and 98th percentile so a few extreme pixels do not wash out the colours.
    pcs = feats[:, :3]
    lo, hi = np.percentile(pcs, [2, 98], axis=0)
    rgb = np.zeros((h * w, 3))
    rgb[inside] = np.clip((pcs - lo) / (hi - lo), 0, 1)
    rgb = rgb.reshape(h, w, 3)
    write_geotiff(a.out / "pca_rgb.tif", (rgb * 255).astype(np.uint8), a.tif)

    # --------------------------------------------------------------------------------
    # STEP 8. Similarity search: which pixels look like the point you chose?
    # --------------------------------------------------------------------------------
    # Cosine similarity (-1 to 1) between each pixel and the reference, in the same PCA space
    # as the zones. The reference is the average of the 3x3 pixels around your point, so a
    # single odd pixel cannot mislead it. Above 0.7 means very similar.
    sim = None
    if a.ref_lat is not None and a.ref_lon is not None:
        r, c = nearest_pixel(a.tif, a.ref_lat, a.ref_lon)
        unit = np.full((h * w, feats.shape[1]), np.nan)
        unit[inside] = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        grid = unit.reshape(h, w, -1)
        ref = np.nanmean(grid[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].reshape(-1, grid.shape[-1]), 0)
        ref /= np.linalg.norm(ref)
        sim = (unit @ ref).reshape(h, w).astype(np.float32)
        write_geotiff(a.out / "similarity.tif", sim, a.tif, nodata=np.nan)
        print(f"Reference pixel row={r}, col={c}; "
              f"{(sim[inside2d] > 0.7).mean() * 100:.1f}% of pixels have similarity > 0.7")

    # --------------------------------------------------------------------------------
    # STEP 9. Quick-look picture of the zones, colour view and similarity map
    # --------------------------------------------------------------------------------
    n = 3 if sim is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes[0].imshow(np.ma.masked_equal(clusters, NODATA), cmap="tab10", interpolation="nearest",
                   vmin=0, vmax=9)
    axes[0].set_title(f"k-means zones (k={k})")
    axes[1].imshow(rgb)
    axes[1].set_title("Embeddings as colour (PCA)")
    if sim is not None:
        im = axes[2].imshow(sim, cmap="viridis", vmin=-1, vmax=1)
        axes[2].plot(c, r, "r+", markersize=14, mew=2)
        axes[2].set_title("Similarity to reference (+)")
        fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(a.out / "overview.png", dpi=150, bbox_inches="tight")
    print(f"Outputs written to {a.out}")


if __name__ == "__main__":
    main()

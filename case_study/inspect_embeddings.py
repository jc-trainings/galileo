"""Inspect the Level 1 embeddings (embeddings_<model>_<id>.npy).

Prints:
  - shape, value range, and a check for NaN/inf values
  - the 128 numbers for one pixel (by lat/lon, or the centre pixel)
  - similarity between two locations
  - how many PCA components explain most of the variation
  - a sanity check: are neighbouring pixels more alike than distant ones?
Writes (optional --geotiff): embeddings.tif, a 128-band GeoTIFF on the same grid
as your input, so QGIS's Identify tool shows all 128 numbers for any pixel you click.

Examples:
    python case_study\\inspect_embeddings.py --emb outputs\\level1\\embeddings_nano_XXXXXXXX.npy --tif AREA
    python case_study\\inspect_embeddings.py --emb ... --tif AREA --lat 0.4747 --lon 34.2306 --lat2 0.47 --lon2 34.24
    python case_study\\inspect_embeddings.py --emb ... --tif AREA --geotiff
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import nearest_pixel, pixel_latlons, write_geotiff  # noqa: E402


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", type=Path, required=True, help="embeddings_*.npy from Level 1")
    p.add_argument("--tif", type=Path, required=True, help="The input .tif used in Level 1")
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--lat2", type=float, default=None)
    p.add_argument("--lon2", type=float, default=None)
    p.add_argument("--geotiff", action="store_true", help="Also write a 128-band GeoTIFF")
    a = p.parse_args()

    emb = np.load(a.emb)
    h, w, d = emb.shape
    flat = emb.reshape(-1, d)
    centred = flat - flat.mean(axis=0)  # same centring as Level 1's similarity map

    # 1. Basic facts
    print(f"File:   {a.emb.name}")
    print(f"Shape:  {h} rows x {w} cols x {d} numbers per pixel "
          f"({h * w:,} pixels, {emb.nbytes / 1e6:.1f} MB)")
    print(f"Values: min {flat.min():.3f}, max {flat.max():.3f}, "
          f"mean {flat.mean():.3f}, std {flat.std():.3f}")
    bad = ~np.isfinite(flat)
    print(f"NaN/inf values: {int(bad.sum())}" + ("  <-- problem!" if bad.any() else "  (good)"))
    dead = int((flat.std(axis=0) < 1e-6).sum())
    print(f"Numbers that never change across pixels: {dead} of {d}")

    # 2. One pixel's embedding
    if a.lat is not None and a.lon is not None:
        r, c = nearest_pixel(a.tif, a.lat, a.lon)
    else:
        r, c = h // 2, w // 2
    lat, lon = pixel_latlons(a.tif).reshape(h, w, 2)[r, c]
    v = emb[r, c]
    print(f"\nPixel row {r}, col {c} (lat {lat:.5f}, lon {lon:.5f}):")
    print("  first 10 numbers:", np.array2string(v[:10], precision=3, suppress_small=True))
    print(f"  length (norm): {np.linalg.norm(v):.3f}")

    # 3. Similarity between two locations
    if a.lat2 is not None and a.lon2 is not None:
        r2, c2 = nearest_pixel(a.tif, a.lat2, a.lon2)
        s_raw = cosine(flat[r * w + c], flat[r2 * w + c2])
        s_cen = cosine(centred[r * w + c], centred[r2 * w + c2])
        print(f"\nSimilarity to pixel row {r2}, col {c2}:")
        print(f"  raw cosine {s_raw:.3f} | centred cosine {s_cen:.3f} "
              "(centred = what similarity.tif shows)")

    # 4. How much of the variation a few directions explain
    pca = PCA(n_components=min(20, d)).fit(flat)
    cum = np.cumsum(pca.explained_variance_ratio_)
    print("\nVariation explained by the top PCA components:")
    for k in (1, 3, 5, 10, 20):
        if k <= len(cum):
            print(f"  top {k:2d}: {cum[k - 1] * 100:5.1f}%")

    # 5. Sanity check: neighbours vs random pairs
    rng = np.random.default_rng(0)
    rows = rng.integers(0, h, 2000)
    cols = rng.integers(0, w - 1, 2000)
    unit = centred / np.linalg.norm(centred, axis=1, keepdims=True)
    u = unit.reshape(h, w, d)
    neigh = np.einsum("nd,nd->n", u[rows, cols], u[rows, cols + 1]).mean()
    other = rng.integers(0, h * w, 2000)
    rand = np.einsum("nd,nd->n", unit[rows * w + cols], unit[other]).mean()
    print("\nAverage centred similarity:")
    print(f"  neighbouring pixels: {neigh:.3f}")
    print(f"  random pairs:        {rand:.3f}")
    print("  (neighbours should be clearly higher: nearby land usually behaves alike)")

    # 6. Optional: 128-band GeoTIFF for clicking in QGIS
    if a.geotiff:
        out = a.emb.with_suffix(".tif")
        write_geotiff(out, emb.astype(np.float32), a.tif)
        print(f"\nWrote {out} ({d} bands). In QGIS: add it, choose the Identify tool, "
              "click a pixel to see all its numbers.")


if __name__ == "__main__":
    main()

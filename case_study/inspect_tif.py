"""List every band of a Galileo export: number, name, month and value range."""
import sys
import numpy as np
import rasterio

path = sys.argv[1]
with rasterio.open(path) as src:
    print(f"File: {path}")
    print(f"Size: {src.width} x {src.height} pixels, {src.count} bands, "
          f"pixel size {src.res[0]:.6f} deg, CRS {src.crs}\n")
    print(f"{'#':>4}  {'band name':<22}{'month':>6}  {'min':>10}  {'mean':>10}  {'max':>10}  {'missing':>7}")
    for i, name in enumerate(src.descriptions, start=1):
        x = src.read(i).astype(np.float64)
        base, _, suffix = (name or "").rpartition("_")
        month = int(suffix) + 1 if suffix.isdigit() and i <= 216 else (1 if i <= 216 else "-")
        print(f"{i:>4}  {name:<22}{month:>6}  {np.nanmin(x):>10.4g}  {np.nanmean(x):>10.4g}  "
              f"{np.nanmax(x):>10.4g}  {np.isnan(x).mean() * 100:>6.2f}%")
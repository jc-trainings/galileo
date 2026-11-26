"""Shared helpers.

The most important function here is `embed_tif_full_inputs`: it is where Level 1 gets its
embeddings. Galileo is NOT downloaded as embeddings from anywhere: the scripts run every
pixel through the pretrained model in data/models/nano/ and keep what comes out.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
from rasterio.warp import transform as warp_transform
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.config import NORMALIZATION_DICT_FILENAME  # noqa: E402
from src.data.dataset import Dataset, DatasetOutput, Normalizer  # noqa: E402
from src.galileo import Encoder  # noqa: E402
from src.masking import MaskedOutput  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: Path) -> Encoder:
    """Load the pretrained Galileo encoder."""
    model = Encoder.load_from_folder(model_dir).to(DEVICE)
    model.eval()  # evaluation mode: the model is only used, never changed
    return model


def galileo_normalizer() -> Normalizer:
    """The same normalization Galileo was pretrained with (std_multiplier=2)."""
    dicts = Dataset.load_normalization_values(REPO_ROOT / "config" / NORMALIZATION_DICT_FILENAME)
    return Normalizer(std=True, normalizing_dicts=dicts, std_multiplier=2)


@torch.no_grad()
def run_encoder(model: Encoder, mo: MaskedOutput) -> np.ndarray:
    """Run a batch through Galileo and return one embedding (128 numbers) per pixel.

    Galileo splits each pixel's inputs into "tokens": one per group of bands per month,
    plus tokens for the bands that do not change over time. The encoder turns each token
    into 128 numbers; we then average all tokens that were not masked out, giving a single
    list of 128 numbers per pixel. That list is the pixel's embedding.
    """
    mo = MaskedOutput(*[t.to(DEVICE) for t in mo])
    out = model(
        mo.space_time_x.float(),
        mo.space_x.float(),
        mo.time_x.float(),
        mo.static_x.float(),
        mo.space_time_mask,
        mo.space_mask,
        mo.time_mask,
        mo.static_mask,
        mo.months.long(),
        patch_size=1,
    )
    # out = (tokens..., masks..., months-ish); average_tokens pools all unmasked tokens
    return model.average_tokens(*out[:-1]).cpu().numpy()


def embed_tif_full_inputs(
    model: Encoder, tif_path: Path, batch_size: int = 256
) -> np.ndarray:
    """Embed every pixel of a Galileo-format GeoTIFF. Used by Level 1.

    Returns an array of shape [H, W, 128]: 128 numbers for every pixel.
    """
    # Step A. Read the export and sort its 233 bands into Galileo's four input groups:
    #   space-time  (each pixel, each month): Sentinel-1, Sentinel-2, NDVI
    #   space       (each pixel, once):       elevation, slope, Dynamic World, WorldCereal
    #   time        (whole tile, each month): rainfall, temperature and other climate values
    #   static      (whole tile, once):       population, location
    # The months are read from the file name, so keep the "dates=" part of the name.
    dso: DatasetOutput = Dataset._tif_to_array(tif_path)

    # Step B. Scale the values exactly as Galileo saw them during pretraining.
    dso = dso.normalize(galileo_normalizer())
    h, w = dso.space_time_x.shape[:2]

    # Step C. Run the pixels through Galileo in batches (one pixel at a time, no neighbours).
    chunks = []
    n_batches = int(np.ceil(h * w / batch_size))
    for batch in tqdm(dso.in_pixel_batches(batch_size=batch_size, window_size=1),
                      total=n_batches, desc="Embedding pixels"):
        mo = MaskedOutput.from_datasetoutput(batch, device=torch.device("cpu"))
        # Step D. Hide the climate ("time") and population/location ("static") inputs
        # (1 = masked). They are the same for every pixel in a 3 km tile, so they cannot
        # tell pixels apart. The original repo's visualizing_embeddings.py does the same.
        mo = mo._replace(
            time_mask=torch.ones_like(mo.time_mask),
            static_mask=torch.ones_like(mo.static_mask),
        )
        # Step E. Galileo turns each pixel into 128 numbers (see run_encoder above).
        chunks.append(run_encoder(model, mo))

    # Step F. Put the pixels back into the image grid: [rows, columns, 128].
    emb = np.concatenate(chunks, axis=0)
    return emb.reshape(h, w, -1)



EDGE_PX = 5


def interior_mask(h: int, w: int, edge: int = EDGE_PX) -> np.ndarray:
    """True for pixels at least `edge` pixels away from the image border."""
    m = np.zeros((h, w), dtype=bool)
    m[edge:h - edge, edge:w - edge] = True
    return m


def tif_profile(tif_path: Path) -> dict:
    with rasterio.open(tif_path) as src:
        return {"crs": src.crs, "transform": src.transform, "height": src.height,
                "width": src.width}


def pixel_latlons(tif_path: Path) -> np.ndarray:
    """Lat/lon of every pixel centre, shape [H*W, 2] (row-major, like the embeddings)."""
    p = tif_profile(tif_path)
    rows, cols = np.meshgrid(np.arange(p["height"]), np.arange(p["width"]), indexing="ij")
    xs, ys = rasterio.transform.xy(p["transform"], rows.ravel(), cols.ravel(), offset="center")
    xs, ys = np.asarray(xs), np.asarray(ys)
    if p["crs"] is not None and p["crs"].to_epsg() != 4326:
        xs, ys = warp_transform(p["crs"], "EPSG:4326", xs, ys)
        xs, ys = np.asarray(xs), np.asarray(ys)
    return np.stack([ys, xs], axis=-1)


def write_geotiff(path: Path, array: np.ndarray, ref_tif: Path,
                  nodata: Optional[float] = None) -> Path:
    """Write [H, W] or [H, W, C] to a GeoTIFF on the same grid as ref_tif."""
    p = tif_profile(ref_tif)
    if array.ndim == 2:
        array = array[..., None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=p["height"], width=p["width"],
        count=array.shape[-1], dtype=array.dtype, crs=p["crs"], transform=p["transform"],
        nodata=nodata, compress="deflate",
    ) as dst:
        for i in range(array.shape[-1]):
            dst.write(array[..., i], i + 1)
    return path


def nearest_pixel(tif_path: Path, lat: float, lon: float) -> tuple:
    latlons = pixel_latlons(tif_path).reshape(tif_profile(tif_path)["height"], -1, 2)
    d = (latlons[..., 0] - lat) ** 2 + (latlons[..., 1] - lon) ** 2
    return np.unravel_index(np.argmin(d), d.shape)


def worldcereal_check(tif_path: Path, prob: np.ndarray, cutoff: float) -> None:
    """Compare a maize probability map [H, W] with the WorldCereal layers in the export.

    WorldCereal is an independent maize map (ESA, made by a different model). The Level 2
    and 3 classifiers do not use it as an input, so agreement is a fair outside check.
    WorldCereal has its own errors, so treat this as a sanity check, not ground truth.
    """
    from sklearn.metrics import roc_auc_score
    with rasterio.open(tif_path) as src:
        names = list(src.descriptions)
        if "maize" not in names:
            return
        wc_maize = src.read(names.index("maize") + 1) > 50
        wc_crop = (src.read(names.index("temporarycrops") + 1) > 50
                   if "temporarycrops" in names else None)
    ok = np.isfinite(prob)
    if wc_maize[ok].all() or not wc_maize[ok].any():
        return
    pred = prob > cutoff
    print("Check against WorldCereal (independent maize map, not used for training):")
    print(f"  AUC of our probability against WorldCereal maize: "
          f"{roc_auc_score(wc_maize[ok], prob[ok]):.2f}")
    print(f"  Of WorldCereal maize pixels, we call maize: {pred[ok & wc_maize].mean() * 100:.0f}%")
    print(f"  Of other pixels, we call maize: {pred[ok & ~wc_maize].mean() * 100:.0f}%")
    if wc_crop is not None:
        print(f"  Of non-cropland pixels (WorldCereal), we call maize: "
              f"{pred[ok & ~wc_crop].mean() * 100:.0f}%")


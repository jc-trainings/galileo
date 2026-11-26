"""Step 1 : export a small Kenyan area from Earth Engine in Galileo format.

The image is built with the repo's own `create_ee_image`, so the band layout is exactly
what Galileo's loader expects: 30-day timesteps of S1, S2, ERA5, TerraClimate and
VIIRS, followed by SRTM, Dynamic World, WorldCereal and LandScan.



Example (3 x 3 km around the CropHarvest Kenya maize test area in Busia):
    python case_study/export_area.py --project YOUR_GEE_PROJECT \
        --lat 0.4747 --lon 34.2306 --size_km 3 --start 2019-01-01 --end 2019-12-31

Then download the .tif from Drive into data/busia/ (keep the file name: Galileo reads
the start month from the "dates=" part of the name).
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import ee

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.earthengine.ee_bbox import EEBoundingBox
from src.data.earthengine.eo import create_ee_image, ee_safe_str


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="Your Earth Engine Cloud project ID")
    p.add_argument("--lat", type=float, default=0.4747)
    p.add_argument("--lon", type=float, default=34.2306)
    p.add_argument("--size_km", type=float, default=3.0, help="Side length of the square")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2019-12-31")
    p.add_argument("--drive_folder", default="galileo_busia")
    a = p.parse_args()

    ee.Initialize(project=a.project)
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    half_m = int(a.size_km * 1000 / 2)
    bbox = EEBoundingBox.from_centre(mid_lat=a.lat, mid_lon=a.lon, surrounding_metres=half_m)
    polygon = bbox.to_ee_polygon()
    name = bbox.get_identifier(start, end)  # min_lat=..._dates=YYYY-MM-DD_YYYY-MM-DD

    img = create_ee_image(polygon, start, end)
    task = ee.batch.Export.image.toDrive(
        image=img.clip(polygon),
        description=ee_safe_str(name)[:100],
        folder=a.drive_folder,
        fileNamePrefix=name,
        region=polygon,
        scale=10,
        maxPixels=1e13,
    )
    task.start()
    print(f"Started export '{name}'.")
    print("Track it at https://code.earthengine.google.com/tasks or with `earthengine task list`.")
    print(f"When done, download {name}.tif from Drive/{a.drive_folder} into data/busia/")


if __name__ == "__main__":
    main()

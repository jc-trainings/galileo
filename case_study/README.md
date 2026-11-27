# Case study: Busia county, Kenya

This case study uses the **Galileo** geospatial foundation model (nano size) on a 3 × 3 km area of
Busia county, western Kenya. It walks through the **three levels of using a foundation model**: first
exploring the landscape without any labels, then mapping maize with labels.

| Level | Question                                 | What happens                                                                                                  | Labels needed         | Script                   |
| ----- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------- | ------------------------ |
| **1** | How is this landscape organised?         | Use the embeddings as they are: group the area into zones of similar land, find places like a chosen location | None                  | `level1_unsupervised.py` |
| **2** | Where is the maize?                      | Keep Galileo fixed; train a small classifier on its embeddings with CropHarvest maize labels                  | Tens to thousands     | `level2_cropharvest.py`  |
| **3** | Can a fine-tuned model map maize better? | Fine-tune Galileo itself for the maize task and compare with Level 2                                          | Hundreds to thousands | `level3_finetune.py`     |

## Files in this folder

| File                      | What it does                                                                                                                |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `export_area.py`          | Exports the study area from Google Earth Engine in Galileo's input format, to your Google Drive                             |
| `level1_unsupervised.py`  | **Level 1**: embeddings, k-means zones, PCA colour view, similarity search                                                  |
| `inspect_embeddings.py`   | Report on the Level 1 embeddings; optional 128-band GeoTIFF                                                                 |
| `level2_cropharvest.py`   | **Level 2**: CropHarvest Kenya maize labels; compares four classifiers and two baselines, label-efficiency curve, maize map |
| `level3_finetune.py`      | **Level 3**: fine-tunes Galileo and compares it with Level 2 on the same data                                               |
| `cropharvest_download.py` | Downloads CropHarvest Levels 2 and 3 call it automatically                                                                  |
| `common.py`               | Shared helpers: load the model, embed pixels, write GeoTIFFs                                                                |

---

## 1. Setup (once)

You need: Windows, macOS or Linux; [Git](https://git-scm.com/downloads);
[Miniforge](https://github.com/conda-forge/miniforge); [QGIS](https://qgis.org) to view maps; and a
Google account with Earth Engine access (step 1a below).

### 1a. Get Earth Engine access

Earth Engine is free for noncommercial use (research, education, non-profits), but every user needs
a **Google Cloud project registered for Earth Engine**. It takes about 5 minutes and access is
immediate. No billing account or payment details are needed for noncommercial use.

1. Sign in to your Google account (a Gmail or work Google account).
2. Open the registration page: https://code.earthengine.google.com/register
3. Choose **Register a Noncommercial or Commercial Cloud project**.
4. Choose **Unpaid usage**, then the project type that fits you (e.g. _Academia & Research_ or
   _Training_). Fill in the short form: your organisation and a one-line description, for example
   "Training workshop on geospatial foundation models".
5. Choose **Create a new Google Cloud Project**. Give it a **Project ID**, e.g. `ee-yourname`.
   The Project ID is lowercase, with no spaces, and must be unique. Write it down: this is your
   `YOUR_PROJECT_ID` in the commands below. (It is the ID, not the display name.)
6. Click **Continue to summary**, check the details and click **Confirm**.
7. Check that it works: open https://code.earthengine.google.com. The Code Editor should load with
   your project selected at the top right.

Already have a Cloud project? On the same registration page you can register it instead of creating
a new one. If you forget your Project ID, it is shown at the top right of the Code Editor and on
https://console.cloud.google.com.

### 1b. Install the case study

On Windows, run every command in **Miniforge Prompt**.

```
cd %USERPROFILE%\Documents
git clone https://github.com/jc-trainings/galileo.git
cd galileo
mamba env create -f environment.yml
conda activate galileo-gfm
earthengine authenticate
```

`earthengine authenticate` opens a browser: sign in with the **same Google account** you used in
step 1a and allow access. Then run `conda activate galileo-gfm` in every new prompt. Check that
Earth Engine works (it should print `1`):

```
python -c "import ee; ee.Initialize(project='YOUR_PROJECT_ID'); print(ee.Number(1).getInfo())"
```

Replace `YOUR_PROJECT_ID` with the Project ID from step 1a (keep the quotes), e.g.
`project='ee-yourname'`.

## 2. Export the study area (shared by all levels)

```
python case_study\export_area.py --project YOUR_PROJECT_ID --lat 0.4747 --lon 34.2306 --size_km 3 --start 2019-01-01 --end 2019-12-31
```

1. Wait until the task shows **Completed** at https://code.earthengine.google.com/tasks.
2. In Google Drive, open **My Drive → galileo_busia** and download the file into `data\busia\`
   **without renaming it**:
   `min_lat=0.4611_min_lon=34.2171_max_lat=0.4883_max_lon=34.2441_dates=2019-01-01_2019-12-31.tif`
3. You don't need to type the long name: type `data\busia\min` and press **Tab**.

The area contains one of the CropHarvest Kenya test areas, so the maps can be checked against field labels.
Below, `AREA` stands for the full quoted path to this file.

## 3. Level 1: explore the landscape (no labels)

No maize labels are used here. The goal is to see what the embeddings already reveal: which pixels
behave alike through the year, how the area splits into zones (fields, homesteads, trees, wetland),
and where else looks like a location you choose with `--ref_lat`/`--ref_lon`. The example uses the
centre of the CropHarvest test area; try a field, a homestead or a wetland you can see on the basemap.

```
python case_study\level1_unsupervised.py --tif AREA --ref_lat 0.4747 --ref_lon 34.2306
python case_study\inspect_embeddings.py --emb outputs\level1\embeddings_nano_XXXXXXXX.npy --tif AREA --geotiff
```

| Output (`outputs\level1\`) | What it is                                                                                                              |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `embeddings_nano_<id>.npy` | The embeddings: 128 numbers for every 10 m pixel                                                                        |
| `clusters.tif`             | Zones of similar pixels from k-means, lightly smoothed (zone 0 is the largest); you name them by comparing with imagery |
| `clusters_unsmoothed.tif`  | The same zones before smoothing, pixel by pixel                                                                         |
| `pca_rgb.tif`              | The embeddings shown as colour; similar colours = similar embeddings                                                    |
| `similarity.tif`           | How much each pixel resembles the reference location (−1 to 1; above 0.7 is very similar)                               |
| `overview.png`             | Quick look at all three                                                                                                 |

How the zones are made: the 128 numbers are standardised and reduced with PCA to the components that
carry 95% of the variation (at most 32), then k-means groups the pixels into 8 zones (`--clusters` to
change; `--smooth 0` turns off the 3 × 3 smoothing). The outer 5 pixels (50 m) of the export are left
out: the Earth Engine export has bad values there (Sentinel-2 about twice as bright, no slope).

**Naming the zones.** The script prints what each zone contains according to Dynamic World, a
separate land-cover map included in the export, for example:

```
zone 2: grass 68%, crops 16%, shrub and scrub 11%
zone 6: trees 96%
```

Dynamic World is itself a model and labels many homestead and bare-soil pixels as "built", so use it
as a hint and confirm on the basemap.

**What Level 1 shows (reference run):** the zones agree with Dynamic World land cover more than twice
as well as the same k-means on the raw 233 satellite bands (normalised mutual information 0.38 against
0.16). Both use the same inputs, so the difference is what Galileo's pretraining adds, with no labels
and no training.

**How many zones?** The script prints a silhouette score for 3 to 10 zones (how cleanly they
separate). In Busia it is highest at 3 and falls slowly: the land changes gradually, from fields to
homesteads to trees, without sharp borders. So there is no single right number; 8 gives zones that
are still easy to name. `--clusters 0` uses the highest silhouette score instead.

**Similarity map:** the reference is the average of the 3 × 3 pixels around your point, so one odd
pixel does not mislead it. Pixels above 0.7 are very similar; in the reference run about 2% of the
area is, mostly other maize fields.

Embedding takes about 5–20 minutes on a laptop CPU (the embeddings are saved, so a rerun takes
seconds). Open the `.tif` files in QGIS over a satellite basemap
(Esri World Imagery: `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}`).

## 4. Level 2: map maize with a simple classifier on the embeddings

```
python case_study\cropharvest_download.py
python case_study\level2_cropharvest.py --tif AREA
```

The first command downloads CropHarvest once (about 1 GB; it resumes if the connection drops).
Level 2 then embeds the 3,940 Kenya training points and does three things:

1. **Label efficiency:** logistic regression and k-NN trained with 20 to 3,940 labels.
2. **Model choice:** four classifiers on the embeddings (logistic regression, k-NN, random forest,
   gradient boosting) are each tuned by cross-validation on the **training points only**, in spatial
   blocks of about 5 km so neighbouring fields do not leak between folds. Each model also gets its own
   maize/not-maize **cut-off**, chosen from the cross-validation predictions (Youden's J), instead of
   assuming 0.5.
3. **Test and map:** all models are scored on the two held-out Busia areas, overall and per area, next
   to two baselines: a traditional random forest on the raw satellite bands (no Galileo) and "call
   every pixel maize". k-NN maps your area (change with `--map_model`; see below).

| Output (`outputs\level2\`) | What it is                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `results_summary.csv`      | F1 and AUC for each method and number of labels                                                                          |
| `label_efficiency.png`     | How the scores change with the number of labels                                                                          |
| `model_comparison.csv`     | Every model and baseline on the test areas: AUC overall and per area, F1 at 0.5, F1 and balanced accuracy at its cut-off |
| `test_regions.png`         | For each test area: ground truth, predicted probability, and maize/not maize at the cut-off                              |
| `maize_probability.tif`    | Maize probability (0–1) for your area                                                                                    |
| `maize_class.tif`          | 1 = maize, 0 = not maize, using the selected model's cut-off                                                             |
| `maize_classifier.joblib`  | The mapping model and its cut-off, to reuse on another area                                                              |

**Reference results** (CropHarvest Kenya, Galileo nano; your numbers should match closely):

| Model                                   | Test AUC, area 1 (54% maize) | Area 2 (79% maize) | Both areas | Balanced accuracy at cut-off |
| --------------------------------------- | ---------------------------- | ------------------ | ---------- | ---------------------------- |
| Galileo + logistic regression           | 0.85                         | 0.53               | 0.77       | 0.70                         |
| **Galileo + k-NN** (used for the map)   | **0.87**                     | **0.76**           | **0.85**   | **0.71**                     |
| Galileo + random forest                 | 0.81                         | 0.55               | 0.75       | 0.69                         |
| Galileo + gradient boosting             | 0.82                         | 0.56               | 0.76       | 0.72                         |
| Random forest on raw bands (no Galileo) | 0.85                         | 0.70               | 0.61       | 0.53                         |
| Call every pixel maize                  | 0.50                         | 0.50               | 0.50       | 0.50                         |

**What to look for:**

- **Cross-validation cannot pick the winner here.** All four classifiers score about 0.96 AUC on the
  training points, but 0.75–0.85 on Busia. Most training points are non-crop land (bush, buildings,
  water), which is easy to tell from maize. In Busia the hard part is maize versus _other crops_
  (cassava, beans, sugarcane), and only about 95 training points near Busia are other crops.
- **k-NN is the best model in both test areas**, so it is the default for the maps. It predicts from
  the most similar labelled fields, which suits a case where the useful labels are a small local group.
  Because this choice was made by looking at the test areas, treat its test score as slightly
  optimistic. `--map_model cv-best` uses the cross-validation choice instead.
- **What Galileo adds is consistency.** Within a single area, the raw-band random forest ranks pixels
  about as well as Galileo k-NN. But its probabilities shift between areas, so with one cut-off its
  balanced accuracy is 0.53 (barely better than guessing), against 0.71 for Galileo k-NN.
- **Few labels go a long way.** Logistic regression trained on only 32 balanced labels scores AUC 0.81,
  better than with all 3,940. More labels mainly add more easy non-crop points.
- The test areas are 64% maize, so "call every pixel maize" already scores F1 ≈ 0.78. Compare models
  on AUC and balanced accuracy.
- In `test_regions.png`, maize fields are found well, but neighbouring non-maize _crop_ fields are
  often called maize too. This is the honest limit of this model and these labels.

**Checking the map of your area.** The two test areas contain only crop fields, so they never show
whether a model wrongly calls homesteads, hedges or trees "maize". The map does. The export includes
**WorldCereal**, an independent maize map from ESA that the classifiers do not use, and the script
compares your map with it. Reference run (k-NN):

```
AUC of our probability against WorldCereal maize: 0.71
Of WorldCereal maize pixels, we call maize: 85%
Of non-cropland pixels (WorldCereal), we call maize: 52%
```

All four classifiers were compared this way too, and k-NN also agrees best with WorldCereal (0.71,
against 0.66–0.69 for the others), which supports choosing it with evidence from outside the test
areas. Two things to tell participants:

- **The probability map is the main result.** Fields, the road and tree lines are clear. The yes/no
  map (`maize_class.tif`) uses the low cut-off learned from all of Kenya (0.18), so it also includes
  much of the land between fields; 62% of the tile is called maize. In QGIS, try showing only
  probabilities above 0.5 for a stricter map (28% maize).
- **WorldCereal is not ground truth either.** It is a model too and often misses small fields, so
  some of the "disagreement" is WorldCereal's error.

## 5. Level 3: map maize by fine-tuning Galileo

```
python case_study\level3_finetune.py --epochs 10 --lrs 1e-4
python case_study\level3_finetune.py --tif AREA
python case_study\level3_finetune.py --map_only --tif AREA

```

- Fine-tuning works on a copy: the original weights in `data\models\nano\` are never changed.

## Notes

- **Inputs:** Levels 2 and 3 embed your map pixels with only the inputs CropHarvest has (Sentinel-1,
  Sentinel-2, NDVI, ERA5 temperature, elevation, location) and mask the rest, so map pixels match the
  training points. Level 1 uses all exported inputs.
- **Export border:** the outer ~5 pixels (50 m) of the Earth Engine export have bad Sentinel-2 values
  and no slope, so every level leaves them out (no-data in the output maps).
- **Timing:** the CropHarvest Kenya arrays cover February 2019 to January 2020, while the export
  covers January to December 2019. Shifting the test pixels by one month changed AUC by only 0.003,
  so the calendar year is fine.
- **Exporting:** the repo's own `EarthEngineExporter` is tied to the original authors' cloud project,
  so `export_area.py` uses the repo's `create_ee_image` but exports to your Google Drive.
- **CropHarvest benchmark:** `eval.py --benchmark cropharvest_kenya` does not run in this repo (it
  requests a classifier the evaluator rejects), so the Level 2 and 3 scripts fit the classifiers themselves.
- **Your own area:** change `--lat`, `--lon`, `--start` and `--end` in step 2. Level 2's classifier is
  trained on Kenya labels, so check results carefully outside western Kenya.

## Optional: a larger Galileo model

The repo includes the **nano** model (about 1 million weights), which runs on any laptop. Larger
models may give better embeddings but are slower. To try the **tiny** model:

```
pip install -U "huggingface_hub[cli]"
hf download nasaharvest/galileo --include "models/tiny/**" --local-dir data
python case_study\level2_cropharvest.py --tif AREA --model data\models\tiny --out outputs\level2_tiny
```

Each model gets its own cached embeddings, and `--model` works the same way for all three levels.

## Optional: NVIDIA GPU (Windows)

The default Windows PyTorch install is CPU-only. For an NVIDIA GPU:

```
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps
```

## Troubleshooting

| Problem                               | Fix                                                                                                       |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `conda` or `mamba` not recognized     | Use **Miniforge Prompt**, not a normal Command Prompt                                                     |
| GDAL or rasterio DLL errors           | rasterio was reinstalled with pip. Run `conda env remove -n galileo-gfm` and create the environment again |
| Earth Engine "project not registered" | Register the project at https://code.earthengine.google.com/register (step 1a)                            |
| Earth Engine "permission denied"      | Sign in to `earthengine authenticate` with the account that owns the project; check the Project ID        |
| Browser does not open on authenticate | Run `earthengine authenticate --auth_mode=notebook` and follow the link it prints                         |
| Export file not in Drive              | Check the task finished; search Drive for `min_lat=0.4611`                                                |
| `need at least 12 timesteps`          | Export a full January–December year                                                                       |
| CropHarvest download stops            | Run `python case_study\cropharvest_download.py` again; it resumes                                         |
| Out of memory                         | Add `--batch_size 64`, or export a smaller `--size_km`                                                    |

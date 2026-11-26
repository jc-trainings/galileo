"""Level 3 — fine-tune Galileo itself for maize, and compare with Level 2.

Question: does changing the model's own weights for maize do better than Level 2?
In Levels 1 and 2 Galileo is frozen. Here its ~1 million weights are trained (on a copy;
the original files in data/models/nano/ are never changed), so the embeddings themselves
adapt to maize.

What the script does, step by step (the same numbers are used in the comments below):
  STEP 1. Load the same CropHarvest Kenya labels as Level 2, and hold back 20% of the
          training points (whole ~5 km blocks) for validation. Validation is used to make
          choices, so the Busia test areas never are.
  STEP 2. Level 2 on the same points, for a fair comparison: frozen Galileo + logistic
          regression, and frozen Galileo + k-NN, each with a cut-off chosen on validation.
  STEP 3. Prepare the starting point (LP-FT, "linear probe, then fine-tune"): copy the
          Level 2 logistic regression into the small "head" on top of Galileo, so
          fine-tuning starts exactly where Level 2 ended.
  STEP 4. Fine-tune Galileo + head, for 15 epochs at each learning rate (1e-4 and 3e-4).
          After every epoch, check validation AUC and keep the best epoch. Epoch 0 (the
          Level 2 starting point) counts too, so Level 3 cannot end up worse than Level 2
          on validation.
  STEP 5. Test the fine-tuned model on the Busia areas next to Level 2 -> comparison.csv
          and save the model and the learning curves  -> finetuned_maize.pt, training_curves.png
  STEP 6. (with --tif) Map your area with the fine-tuned model and check it against
          WorldCereal -> maize_probability_finetuned.tif/.png, maize_class_finetuned.tif
          Already trained? --map_only --tif <file> loads finetuned_maize.pt and only maps.

How much of the encoder to train (--train_blocks):
  -1  all weights (full fine-tuning; the default)
   k  only the last k transformer blocks + final norm (partial fine-tuning)
   0  head only: a linear probe trained with gradients (sanity check vs Level 2)

Examples:
    python case_study/level3_finetune.py                        # full fine-tuning
    python case_study/level3_finetune.py --train_blocks 2       # last 2 blocks only
    python case_study/level3_finetune.py --n_labels 64          # see what few labels do
    python case_study/level3_finetune.py --tif data/busia/<file>.tif

Nano trains fine on CPU (minutes per learning rate). For tiny or base, use a GPU.
"""

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEVICE,
    REPO_ROOT,
    galileo_normalizer,
    interior_mask,
    load_model,
    pixel_latlons,
    tif_profile,
    worldcereal_check,
    write_geotiff,
)
from level2_cropharvest import (  # noqa: E402
    CH_TIMESTEPS,
    SEED,
    balanced_subset,
    embed_cropharvest_array,
    load_kenya,
    make_cv,
    make_models,
    spatial_blocks,
    test_scores,
    youden_threshold,
    tif_to_cropharvest_array,
)
from src.galileo import adjust_learning_rate  # noqa: E402
from src.masking import MaskedOutput  # noqa: E402


# ---------------------------------------------------------------------------
# The network: Galileo + a small "head"
#
# Galileo turns a pixel into 128 numbers (exactly as in Levels 1 and 2). The head is one
# linear layer that turns those 128 numbers into a maize score. In Level 2 only a classifier
# like this is trained; here Galileo's own weights are trained too.
# ---------------------------------------------------------------------------
class EncoderWithHead(nn.Module):
    def __init__(self, encoder, train_blocks: int = -1):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)  # never modify the pretrained weights
        self.head = nn.Linear(self.encoder.embedding_size, 1)
        self._set_trainable(train_blocks)

    def _set_trainable(self, train_blocks: int):
        """Choose which weights may change (--train_blocks): all, the last k blocks, or none
        (head only). Weights that may not change are "frozen"."""
        enc = self.encoder
        if train_blocks < 0:  # full fine-tuning
            for p in enc.parameters():
                p.requires_grad = True
            return
        for p in enc.parameters():
            p.requires_grad = False
        if train_blocks > 0:  # unfreeze only the last k blocks + final norm
            for block in list(enc.blocks)[-train_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in enc.norm.parameters():
                p.requires_grad = True

    def forward(self, mo: MaskedOutput) -> torch.Tensor:
        """Pixels -> Galileo -> 128 numbers (averaged tokens) -> head -> one maize score."""
        out = self.encoder(
            mo.space_time_x, mo.space_x, mo.time_x, mo.static_x,
            mo.space_time_mask, mo.space_mask, mo.time_mask, mo.static_mask,
            mo.months.long(), patch_size=1,
        )
        emb = self.encoder.average_tokens(*out[:-1])  # [B, D] — same pooling as Level 2
        return self.head(emb).squeeze(-1)  # one logit per sample


def to_tensors(converter, x, latlons) -> MaskedOutput:
    """Raw CropHarvest arrays -> normalized, masked Galileo inputs (repo's own function)."""
    return converter.cropharvest_array_to_normalized_galileo(
        x, latlons, start_month=converter.start_month, timesteps=CH_TIMESTEPS)


def batches(mo: MaskedOutput, idx: np.ndarray, batch_size: int):
    for i in range(0, len(idx), batch_size):
        b = torch.as_tensor(idx[i:i + batch_size])
        yield b, MaskedOutput(*[t[b].to(DEVICE) for t in mo])


@torch.no_grad()
def predict(net: EncoderWithHead, mo: MaskedOutput, batch_size: int = 512) -> np.ndarray:
    net.eval()
    out = [torch.sigmoid(net(b)).cpu().numpy()
           for _, b in batches(mo, np.arange(len(mo.months)), batch_size)]
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Training for one learning rate (used in STEP 4)
#
# One epoch = one pass through all training points, in small batches. For each batch:
#   predict -> measure the error (loss) -> nudge the weights to reduce it.
# The learning rate sets how big each nudge is.
# ---------------------------------------------------------------------------
def finetune(encoder, train_blocks, mo_tr, y_tr, mo_va, y_va, lr, epochs, batch_size,
             head_init=None):
    torch.manual_seed(SEED)
    net = EncoderWithHead(encoder, train_blocks).to(DEVICE)
    if head_init is not None:  # LP-FT: start from the Level 2 linear solution
        w, b = head_init
        with torch.no_grad():
            net.head.weight.copy_(torch.as_tensor(w, dtype=torch.float32).view(1, -1))
            net.head.bias.copy_(torch.as_tensor([b], dtype=torch.float32))
    # The optimiser (AdamW) changes the trainable weights; weight decay keeps them from
    # drifting far from their starting values.
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    # Loss: how wrong the predictions are. Maize errors count more, because maize is only
    # 7% of the training points (otherwise "never maize" would look good).
    pos = max(int(y_tr.sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(  # balance classes, like class_weight="balanced"
        pos_weight=torch.tensor((len(y_tr) - pos) / pos, device=DEVICE))
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32)
    rng = np.random.default_rng(SEED)
    warmup = max(1, int(epochs * 0.1))
    steps_per_epoch = int(np.ceil(len(y_tr) / batch_size))

    # Epoch 0 = the starting point. With LP-FT this is the Level 2 solution, so if
    # fine-tuning only makes validation worse, that solution is kept.
    val0 = roc_auc_score(y_va, predict(net, mo_va))
    best = {"val_auc": val0, "epoch": 0, "state": copy.deepcopy(net.state_dict())}
    history = [{"lr": lr, "epoch": 0, "train_loss": np.nan, "val_auc": val0}]
    print(f"  lr={lr:g} epoch   0/{epochs}  (starting point)        val AUC {val0:.4f}")
    for epoch in range(epochs):
        net.train()
        losses = []
        for step, (b_idx, mo_b) in enumerate(batches(mo_tr, rng.permutation(len(y_tr)),
                                                      batch_size)):
            # learning rate: rises gently in the first epoch, then slowly falls (cosine)
            adjust_learning_rate(opt, epoch + step / steps_per_epoch, warmup, epochs,
                                 lr, lr / 100)
            loss = loss_fn(net(mo_b), y_tr_t[b_idx].to(DEVICE))  # predict and measure error
            opt.zero_grad()
            loss.backward()                                       # work out each nudge
            torch.nn.utils.clip_grad_norm_(params, 1.0)           # cap very large nudges
            opt.step()                                            # nudge the weights
            losses.append(loss.item())
        # After each epoch: how well does the model do on the held-back validation points?
        val_auc = roc_auc_score(y_va, predict(net, mo_va))
        history.append({"lr": lr, "epoch": epoch + 1, "train_loss": np.mean(losses),
                        "val_auc": val_auc})
        print(f"  lr={lr:g} epoch {epoch + 1:3d}/{epochs}  "
              f"train loss {np.mean(losses):.4f}  val AUC {val_auc:.4f}")
        if val_auc > best["val_auc"]:  # keep the best epoch (early stopping)
            best = {"val_auc": val_auc, "epoch": epoch + 1,
                    "state": copy.deepcopy(net.state_dict())}
    net.load_state_dict(best["state"])
    return net, best, history


# ---------------------------------------------------------------------------
# Mapping (used in STEP 6, and on its own with --map_only)
# ---------------------------------------------------------------------------
def map_area(net, converter, tif: Path, cutoff: float, out: Path):
    """Maize probability and yes/no maps of an export, with the fine-tuned model."""
    print(f"\nMapping {tif.name}")
    # Describe every pixel the same way as the labelled points (CropHarvest inputs only)
    mo_map = to_tensors(converter, tif_to_cropharvest_array(tif), pixel_latlons(tif))
    prof = tif_profile(tif)
    # The fine-tuned network gives each pixel a maize probability (0-1)
    prob = predict(net, mo_map).reshape(prof["height"], prof["width"])
    prob[~interior_mask(*prob.shape)] = np.nan  # the export's bad outer border
    write_geotiff(out / "maize_probability_finetuned.tif", prob.astype(np.float32), tif,
                  nodata=np.nan)
    # Yes/no map at the cut-off chosen on validation (255 = no-data border)
    cls = np.where(np.isfinite(prob), prob > cutoff, 255).astype(np.uint8)
    write_geotiff(out / "maize_class_finetuned.tif", cls, tif, nodata=255)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(prob, cmap="RdYlGn", vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[0], label="P(maize)")
    axes[0].set_title("Maize probability (fine-tuned Galileo)")
    axes[1].imshow(np.ma.masked_equal(cls, 255), cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_title(f"Maize (probability > {cutoff:.2f})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "maize_probability_finetuned.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {(prob[np.isfinite(prob)] > cutoff).mean() * 100:.1f}% of pixels predicted "
          f"maize at cut-off {cutoff:.2f}")
    # Compare with WorldCereal's maize map (independent, not used for training)
    worldcereal_check(tif, prob, cutoff)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=REPO_ROOT / "data/models/nano")
    p.add_argument("--train_blocks", type=int, default=-1,
                   help="-1 = all, k = last k blocks, 0 = head only")
    p.add_argument("--lrs", type=float, nargs="+", default=None,
                   help="Learning rates to try (default: 1e-4 and 3e-4 when the encoder "
                        "is trained, 1e-3..1e-1 for head only)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lpft", type=int, default=1,
                   help="1 = start from the Level 2 logistic regression (LP-FT); 0 = random head")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_labels", type=int, default=0,
                   help="Use only this many training labels (0 = all)")
    p.add_argument("--tif", type=Path, default=None, help="Optional area to map")
    p.add_argument("--map_only", action="store_true",
                   help="Skip training: load the saved finetuned_maize.pt from --out and map --tif")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/level3")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.lrs is None:
        # a lone linear head needs far larger steps than pretrained encoder weights
        a.lrs = [1e-3, 1e-2, 3e-2, 1e-1] if a.train_blocks == 0 else [1e-4, 3e-4]
    print(f"Device: {DEVICE}")

    encoder = load_model(a.model)  # pretrained Galileo; fine-tuning works on a copy of it
    ev, (x_tr_all, ll_tr_all, y_tr_all), (x_te, ll_te, y_te, region) = load_kenya(
        galileo_normalizer())

    # --map_only: reuse a model fine-tuned earlier, no training (saves ~30 minutes)
    if a.map_only:
        if a.tif is None:
            raise SystemExit("--map_only needs --tif (the area to map)")
        ckpt_path = a.out / "finetuned_maize.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"{ckpt_path} not found: run level3_finetune.py once without "
                             "--map_only first")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        net = EncoderWithHead(encoder, ckpt["train_blocks"]).to(DEVICE)
        net.load_state_dict(ckpt["state_dict"])
        print(f"Loaded {ckpt_path} (lr={ckpt['lr']:g}, epoch {ckpt['epoch']}, "
              f"cut-off {ckpt['cutoff']:.2f})")
        map_area(net, ev, a.tif, ckpt["cutoff"], a.out)
        print(f"\nOutputs written to {a.out}")
        return

    # --------------------------------------------------------------------------------
    # STEP 1. Labels, and a validation set held back from training
    # --------------------------------------------------------------------------------
    # --n_labels: optionally use only a few labels, to see how fine-tuning copes with less.
    if a.n_labels:
        keep = balanced_subset(y_tr_all, a.n_labels, np.random.default_rng(SEED))
        x_tr_all, ll_tr_all, y_tr_all = x_tr_all[keep], ll_tr_all[keep], y_tr_all[keep]
    # Validation = one fold of the same spatial split as Level 2 (~20% of points, whole
    # ~5 km blocks), so choices are made on areas the model did not train on
    folds, cv_kind = make_cv(y_tr_all, spatial_blocks(ll_tr_all))
    i_tr, i_va = folds[0]
    print(f"Validation split: {cv_kind}")
    y_tr, y_va = y_tr_all[i_tr], y_tr_all[i_va]
    print(f"Train {len(y_tr)} / validation {len(y_va)} / test {len(y_te)}")

    # Prepare the points as Galileo inputs (same as Level 2: CropHarvest inputs only)
    mo_tr = to_tensors(ev, x_tr_all[i_tr], ll_tr_all[i_tr])
    mo_va = to_tensors(ev, x_tr_all[i_va], ll_tr_all[i_va])
    mo_te = to_tensors(ev, x_te, ll_te)

    # --------------------------------------------------------------------------------
    # STEP 2. Level 2 on the same points, for a fair comparison
    # --------------------------------------------------------------------------------
    # Embed train, test and validation points with frozen Galileo, then train logistic
    # regression and k-NN on the same 80% the fine-tuning uses. Each gets a cut-off chosen
    # on the validation points.
    e_tr = embed_cropharvest_array(encoder, ev, x_tr_all[i_tr], ll_tr_all[i_tr], desc="Train")
    e_te = embed_cropharvest_array(encoder, ev, x_te, ll_te, desc="Test")
    e_va = embed_cropharvest_array(encoder, ev, x_tr_all[i_va], ll_tr_all[i_va], desc="Val")
    baseline = make_models(len(y_tr))["LogisticRegression"].fit(e_tr, y_tr)
    cut_l2 = youden_threshold(y_va, baseline.predict_proba(e_va)[:, 1])
    region = np.asarray(region)
    regions = sorted(set(region))

    def scores(prob, cut):  # overall scores plus AUC in each test area
        return {**test_scores(y_te, prob, cut),
                **{f"test_auc_area{i + 1}": roc_auc_score(y_te[region == rg], prob[region == rg])
                   for i, rg in enumerate(regions)}}

    rows = [{"method": "Level 2: frozen Galileo + logistic regression", "cutoff": cut_l2,
             **scores(baseline.predict_proba(e_te)[:, 1], cut_l2)}]
    # k-NN with the settings Level 2's cross-validation chose on the full Kenya data
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(
        n_neighbors=40, weights="distance", metric="cosine")).fit(e_tr, y_tr)
    cut_knn = youden_threshold(y_va, knn.predict_proba(e_va)[:, 1])
    rows.append({"method": "Level 2: frozen Galileo + k-NN", "cutoff": cut_knn,
                 **scores(knn.predict_proba(e_te)[:, 1], cut_knn)})

    # --------------------------------------------------------------------------------
    # STEP 3. Start from the Level 2 answer (LP-FT)
    # --------------------------------------------------------------------------------
    # Logistic regression is itself a linear layer: 128 weights and a bias. We copy them
    # (after undoing the scaling step) into the head, so before any training the network
    # gives exactly the Level 2 predictions. --lpft 0 starts from a random head instead.
    head_init = None
    if a.lpft:
        scaler, logreg = baseline.steps[0][1], baseline.steps[-1][1]
        w = logreg.coef_[0] / scaler.scale_
        b = float(logreg.intercept_[0] - np.sum(logreg.coef_[0] * scaler.mean_ / scaler.scale_))
        head_init = (w, b)
        print("LP-FT: fine-tuning starts from the Level 2 logistic-regression solution")

    # --------------------------------------------------------------------------------
    # STEP 4. Fine-tune at each learning rate; keep the best epoch on validation
    # --------------------------------------------------------------------------------
    # How many weights will change: all ~1 million by default (--train_blocks -1).
    n_train = sum(p.numel() for p in EncoderWithHead(encoder, a.train_blocks).parameters()
                  if p.requires_grad)
    print(f"Trainable parameters: {n_train:,} "
          f"(encoder has {sum(p.numel() for p in encoder.parameters()):,})")
    history, best_net, best = [], None, {"val_auc": -1.0}
    for lr in a.lrs:
        print(f"\nFine-tuning with lr={lr:g}")
        net, b, h = finetune(encoder, a.train_blocks, mo_tr, y_tr, mo_va, y_va, lr,
                             a.epochs, a.batch_size, head_init=head_init)
        history += h
        if b["val_auc"] > best["val_auc"]:
            best, best_net = {**b, "lr": lr}, net
    pd.DataFrame(history).to_csv(a.out / "training_history.csv", index=False)
    print(f"\nChosen: lr={best['lr']:g}, epoch {best['epoch']} "
          f"(validation AUC {best['val_auc']:.4f})")

    # --------------------------------------------------------------------------------
    # STEP 5. Test on the Busia areas, next to Level 2, and save
    # --------------------------------------------------------------------------------
    # The cut-off is chosen on validation, as for Level 2. comparison.csv has one row each
    # for Level 2 logistic regression, Level 2 k-NN and Level 3.
    prob_te = predict(best_net, mo_te)
    cut_l3 = youden_threshold(y_va, predict(best_net, mo_va))
    mode = {-1: "all weights", 0: "head only"}.get(a.train_blocks,
                                                     f"last {a.train_blocks} blocks")
    rows.append({"method": f"Level 3: fine-tuned ({mode}{', LP-FT' if a.lpft else ''})",
                 "cutoff": cut_l3, **scores(prob_te, cut_l3)})
    if best["epoch"] == 0:
        print("Note: no epoch beat the starting point on validation, so the Level 2 "
              "solution was kept (fine-tuning did not help with these labels).")
    res = pd.DataFrame(rows).round(4)
    res["n_train_labels"] = len(y_tr)
    res.to_csv(a.out / "comparison.csv", index=False)
    print("\nTest-set comparison:\n", res.to_string(index=False))
    torch.save({"state_dict": best_net.state_dict(), "train_blocks": a.train_blocks,
                "lr": best["lr"], "epoch": best["epoch"], "cutoff": cut_l3},
               a.out / "finetuned_maize.pt")

    # training_curves.png: training loss (should fall) and validation AUC (should rise, then
    # level off; if it falls while the loss keeps falling, the model is memorising).
    hist = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for lr, g in hist.groupby("lr"):
        axes[0].plot(g["epoch"], g["train_loss"], label=f"lr={lr:g}")
        axes[1].plot(g["epoch"], g["val_auc"], label=f"lr={lr:g}")
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation AUC")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(a.out / "training_curves.png", dpi=150)

    # --------------------------------------------------------------------------------
    # STEP 6. (with --tif) Map your area with the fine-tuned model
    # --------------------------------------------------------------------------------
    # Same as Level 2 step 8: describe the pixels like the labelled points, predict a maize
    # probability for each, set the border to no-data, apply the cut-off, check WorldCereal.
    if a.tif is not None:
        map_area(best_net, ev, a.tif, cut_l3, a.out)

    print(f"\nOutputs written to {a.out}")


if __name__ == "__main__":
    main()

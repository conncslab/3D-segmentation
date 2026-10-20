# CPSAM3D: 3D instance segmentation with cellpose-SAM

A pipeline for segmenting objects such as cells or nuclei in 3D fluorescence images (z-stacks).
It trains a model on human-corrected masks (human-in-the-loop), runs the trained model over the whole volume,
and links the per-slice results into 3D objects (stitching) to produce the final 3D masks.

```
raw volume ──► ① HITL training (hitl.py) ──► ② whole-volume inference (3D_segmentation.py) ──► ③ 3D stitch (stitch.py) ──► 3D masks
               (correct views ↔ retrain)        per-slice 2D masks                                linked along z
```

## Model

- **cellpose-SAM**: uses the `cpsam_v2` weights (SAM ViT-L based) trained by the cellpose team, as is. It runs on each
  slice separately and predicts the flows (dY, dX) and the cell probability. The encoder stays frozen during training.
- **3D U-Net**: looks at the raw image and the cellpose-SAM output together across z, and adds a correction to that output.
  Its last layer is zero-initialized, so before training the model gives exactly cellpose-SAM's result and departs from it
  only as it learns.
- **Post-processing**: on every slice, flows are followed the cellpose way to form 2D objects; at the end, pieces on
  adjacent slices are linked 1:1 by IoU into 3D objects (the same rule as cellpose's `stitch3D`).

## Requirements

- Python 3.12, PyTorch, cellpose 4.2.x, GPU

### Installation

```bash
conda create -n cpsam3d python=3.12
conda activate cpsam3d

# Install the PyTorch build for your GPU's CUDA version (see https://pytorch.org). Example: CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install cellpose==4.2.1.1 edt numba
```

The `cpsam_v2` weights (about 1.2 GB) are downloaded automatically to `~/.cellpose/models/` on the first run.

## Code layout

| File | Role |
|---|---|
| `src/config.py` | All settings (data paths, pixel size, training and inference parameters) |
| `src/hitl.py` | Human-in-the-loop: training + extracting and predicting a new view |
| `src/3D_segmentation.py` | Block-wise inference over the whole volume with the trained model, assembled into one per-slice mask |
| `src/stitch.py` | Links the per-slice masks into 3D objects |
| `src/model/` | CPSAM3D (cellpose-SAM + 3D U-Net) |
| `src/step1~6_*.py`, `src/utils/` | Data access, preprocessing, flow computation, training loop, inference and post-processing |

## 1. Data preparation

### Raw volume

Use one of these two formats:

- **One (Z, H, W) tif**: a single multi-page tif (one page per slice)
- **One tif per slice**: several (H, W) tifs numbered from z = 0, such as `xxx_z0.tif`, `xxx_z1.tif`, ...

Requirements:

- The data type must be uint16 (or uint8). The normalization range is computed from a 0–65535 histogram.
- The tifs must be **uncompressed**. The code reads only the parts it needs through `tifffile.memmap` instead of loading
  the whole volume into memory. Saving without a compression option, as in `tifffile.imwrite(path, volume)`, is enough.

### Configuration

Open `src/config.py` and set the items below for your data. The docstring at the top lists the items you must change
and the ones to check depending on your data.

| Item | Description |
|---|---|
| `PROJECT_NAME` | Prefix of the output file names |
| `DATA_DIR` | Folder holding the raw volume |
| `VOLUME_FILE` | File name of the single tif, or `None` for one file per slice |
| `Z_FILE_PATTERN` | File name pattern for one file per slice; `{z}` is replaced by 0, 1, 2, ... (e.g. `'my_volume_z{z}.tif'`) |
| `HITL_DIR` | Folder where the views to correct and the trained model are stored |
| `XY_PIXEL_UM`, `Z_STEP_UM` | xy pixel size and z step (µm) |
| `RESCALE_FACTOR` | xy downscale before the model. cellpose-SAM works in 256×256 tiles, so objects must fit well inside one tile: 1 for small objects, 2, 4, 8, ... for very large ones |
| `REVIEW_SIZE_XY`, `REVIEW_SIZE_Z` | Size of one view you correct at a time (xy at least `256 × RESCALE_FACTOR`) |
| `PATCH_SIZE_Z` | Number of z slices the model sees at once; at most the number of slices of the data and at most `REVIEW_SIZE_Z` |
| `MIN_SIZE_PX` | Slice pieces smaller than this are deleted (in raw pixels). Keep it below the real cross-sections of your objects |

### Intensity normalization

Each volume is normalized with the 1–99 percentile range of the whole volume. On the first run the whole volume is
scanned once, and the range is saved as `global_norm.json` and reused afterwards.

- `DATA_DIR/global_norm.json`: range of the volume to predict
- `HITL_DIR/global_norm.json`: range of the volume the training views were cut from

If the training views come from the volume you predict (`DATA_DIR`), there is nothing to do. If you train on views cut
from another volume, put that volume's `global_norm.json` (`{"p_low": ..., "p_high": ...}`) in `HITL_DIR`.
Otherwise it is computed from the `DATA_DIR` volume, and a range that doesn't match the training views is used.

## 2. Training (human-in-the-loop)

Training repeats "cut a view and predict it → a human corrects it → retrain on every view corrected so far".

### First run

```bash
python src/hitl.py <x> <y> [z]
```

- `x`, `y` are pixel coordinates in the raw volume (x = column, y = row). Pick the spot you want to correct in a viewer
  (napari, Neuroglancer, ...).
- The `REVIEW_SIZE_XY` grid cell that coordinate falls in is cut out as the view. With `z`, the `REVIEW_SIZE_Z` range
  that z falls in is used; without it, the view starts at the first slice.
- With no corrected view yet, the view is predicted by cellpose-SAM as is, without training.

These files appear in `HITL_DIR` (named after the grid row and column, such as `view_r012_c034`):

| File | Content |
|---|---|
| `view_rXXX_cXXX_img.tif` | The cut-out raw image (Z, H, W) |
| `view_rXXX_cXXX_masks.tif` | Per-slice masks predicted by the model |
| `view_rXXX_cXXX_flow_vis.tif` | Color preview of the predicted flows |

### Correcting masks

Open `_masks.tif` in any editor (napari, ...), correct it, and save it in the same folder as
**`view_rXXX_cXXX_masks_modified.tif`**.

- Format: (Z, H, W) integer labels of the same size as `_img.tif`, background 0
- Per-slice 2D masks are enough. The same object may have different ids on different slices (ids are renumbered per
  slice for training).
- Only views with a `_masks_modified.tif` count as corrected and go into training. The original `_masks.tif` is never
  overwritten.

### Retraining + next view

```bash
python src/hitl.py <next x> <next y>
```

1. Retrains the model **from scratch** on every view corrected so far (it does not continue from the previous round's
   model; this follows cellpose's human-in-the-loop recommendation and weighs every view equally).
2. Saves the model to `HITL_DIR/models/main/model.pth` and the log to `train_log.txt`.
3. Cuts the view at the new coordinate and predicts it with the trained model.

Repeat "correct → retrain" until the results are good enough.

### If you already have ground-truth masks

Put existing ground truth in `HITL_DIR` with these names and it is used for training right away:

```
view_<name>_img.tif              # (Z, H, W) raw image
view_<name>_masks_modified.tif   # (Z, H, W) ground-truth labels
```

Then `python src/hitl.py <x> <y>` trains on these views and predicts the new view at the coordinate.

> **Caution**: `hitl.py` saves the newly cut view as `view_r{row}_c{col}_img.tif` and overwrites a file of the same name.
> If the name of a ground-truth view you added matches this pattern, its raw image can be replaced, so keep the names distinct.

### Training settings and monitoring

- Amount of training: every epoch draws `AREA_RATIO` 256×256 crops per view (= the number of tiles in one view).
  The total number of steps is `N_EPOCHS × number of views × AREA_RATIO ÷ BATCH_SIZE`.
- By default only the 3D U-Net and cellpose-SAM's last output layer (readout) are trained; the ViT encoder stays frozen
  (`UNFREEZE_LAST_N_BLOCKS = 0`). Raise this value to also train the encoder's last few blocks (needs more GPU memory).
- `train_log.txt` shows the loss every 10 epochs and the AP@0.5 on the training views every `EVAL_EVERY` epochs.
  This AP is measured on the training data itself, so it only checks that training works; it is not generalization performance.
- `ZSMOOTH_BOUNDARY_*` is an extra loss that keeps the predictions of adjacent slices alike. Both weights 0 turn it off.
- Training writes `*_img_flows.tif` files (cached training flows) into `HITL_DIR`. They can be deleted.

## 3. Whole-volume inference

```bash
python src/3D_segmentation.py
```

- Runs `HITL_DIR/models/main/model.pth` over the whole volume block by block (`BLOCK_SIZE`, `BLOCK_SIZE_Z`).
  Each block is read with a margin around it (`MARGIN`, `MARGIN_Z`), so objects at block borders are segmented with
  their surroundings.
- Block results are saved in `SEGMENTATION_DIR` (default: `DATA_DIR/segmentation`), and finished blocks are recorded
  in `progress.json`. If it stops halfway, running it again resumes where it left off.
- At the end the blocks are combined into **`<PROJECT_NAME>_predmasks2d_combined.tif`**: (Z, H, W) uint32 per-slice
  masks, not yet linked along z. Objects cut by an xy block border are merged on every slice.

Environment variables override the defaults:

| Environment variable | Purpose |
|---|---|
| `CPSAM3D_CKPT_OVERRIDE` | Run inference with another checkpoint |
| `CPSAM3D_OUT_DIR_OVERRIDE` | Save the results in another folder |

> **Caution**
> - Before rerunning after changing the model or the settings, delete `progress.json` and the `block_*` files in
>   `SEGMENTATION_DIR`. Otherwise the old block results are reused.
> - The final assembly step holds the whole result in memory (Z × H × W × 4 bytes). If that is larger than your RAM,
>   the assembly step runs out of memory.

## 4. 3D stitch

```bash
python src/stitch.py <per-slice masks.tif> [output.tif] [--threshold 0.25]
```

Example:

```bash
python src/stitch.py data/my_volume/segmentation/my_project_predmasks2d_combined.tif
```

- Pieces on adjacent slices are linked 1:1 when their IoU is at least `STITCH_THRESHOLD` (default 0.25) and each is
  the other's best match; each chain of links becomes one 3D object. This is the same rule as cellpose's `stitch3D`.
- Without an output path, the result is saved next to the input as **`<input name>_stitched.tif`**: (Z, H, W) uint32
  3D labels, numbered from 1 in order of first appearance.
- Input ids may be numbered per slice or be unique over the whole volume. It also works on the view masks of `hitl.py`.
- Slices are read in bands and the result is written straight to disk, so volumes much larger than RAM can be processed.
  A compressed input tif, however, can't be opened as a memmap and is loaded into memory entirely.
- One difference from cellpose's `stitch3D`: in cellpose, when an empty slice comes first, ids on the following slices
  can repeat ids of earlier slices and fuse unrelated objects; that cannot happen here.


## Common settings to adjust

| Situation | Setting |
|---|---|
| Objects too big for the 256 px tiles | Raise `RESCALE_FACTOR` (and `REVIEW_SIZE_XY` with it) |
| Many small noise pieces / real small cross-sections get deleted | `MIN_SIZE_PX` |
| Many odd-shaped pieces remain / too many get deleted | `FLOW_THRESHOLD` (lower is stricter; cellpose default 0.4) |
| Objects don't link well along z / different objects get linked | `STITCH_THRESHOLD` (`stitch.py --threshold`) |
| Out of GPU memory | Lower `BATCH_SIZE`, `UNFREEZE_LAST_N_BLOCKS`; for inference, lower `BLOCK_SIZE` |
| Objects cut at block borders | Set `MARGIN` to at least one object diameter |

`PATCH_SIZE_Z`, `XY_PIXEL_UM`, `Z_STEP_UM` and `RESCALE_FACTOR` set the U-Net's z pooling and the input scale of the
model. After changing them, an earlier checkpoint still loads but no longer matches the conditions it was trained
under, so retrain.

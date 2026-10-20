"""Settings of the pipeline: hitl.py (human-in-the-loop training), 3D_segmentation.py (whole-volume per-slice
segmentation), the per-slice mask computation inside inference (step6) and stitch.py (3D linking).

MUST EDIT for new data -- these follow the data, not the code:
  DATA_DIR, VOLUME_FILE /   where the raw volume is: one combined (Z,H,W) tif, or one file per z-slice
  Z_FILE_PATTERN
  HITL_DIR                  where the human-in-the-loop review regions and the trained model are kept
  XY_PIXEL_UM, Z_STEP_UM    physical voxel size of the raw data (um). They set the model's z/xy ratio
                            (Z_XY_ANISOTROPY), which shapes the U-Net.
  RESCALE_FACTOR            xy downscale before the model, from the object size: cellpose-SAM works in 256 x 256 px
                            tiles, so objects must fit well inside one. Try a few factors on one region and keep the
                            one cellpose-SAM segments best (1 = no rescale).
  REVIEW_SIZE_XY            xy size (raw px) of one region a human corrects per HITL round; >= 256 * RESCALE_FACTOR.
  REVIEW_SIZE_Z             z depth (slices) of that region; >= PATCH_SIZE_Z.
  PATCH_SIZE_Z              z depth (slices) the model sees at once; long enough to see an object above and below a
                            contact (a few object diameters), at most the number of slices of the data.

CHECK when the objects' size, density or the GPU change:
  MIN_TRAIN_MASKS                       objects a training crop must hold (object density at the model's resolution)
  ZSMOOTH_BOUNDARY_DILATE_PX            width of the outline band of the z-smoothness loss (object size)
  MIN_SIZE_PX                           smallest piece kept on a slice (object cross-section area)
  MARGIN, MARGIN_Z                      block overlap of whole-volume inference: at least one object across / deep
  BATCH_SIZE, UNFREEZE_LAST_N_BLOCKS    GPU memory
"""
import os


# Project folder = the folder holding src/ (absolute, so the scripts run from any working directory). Layout:
#
#   <DATA_DIR>/                                     the raw volume (VOLUME_FILE, or one Z_FILE_PATTERN file per slice)
#   ├── global_norm.json                            DATA_NORM_CACHE: intensity normalization range of this volume (computed once)
#   └── segmentation/                               SEGMENTATION_DIR
#       ├── block_*_predmasks2d.tif, progress.json  3D_segmentation.py: per-block results (a rerun resumes)
#       └── <PROJECT_NAME>_predmasks2d_combined.tif 3D_segmentation.py: per-slice masks of the whole volume
#   <HITL_DIR>/                                     hitl.py
#   ├── view_r000_c000_img.tif                      one review region of a raw volume
#   ├── view_r000_c000_masks.tif                    the model's per-slice masks of it
#   ├── view_r000_c000_masks_modified.tif           your corrected masks -- saving this file makes the region "reviewed"
#   ├── view_*_flow_vis.tif, view_*_img_flows.tif   flow preview / cached training targets
#   ├── global_norm.json                            TRAIN_NORM_CACHE: normalization range of the volume the views were cut from
#   └── models/main/                                MAIN_MODELS_DIR: model.pth (MAIN_CHECKPOINT_PATH), train_log.txt
PROJECT_NAME = 'colon'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'colon')
Z_FILE_PATTERN = 'mosaic_DAPI_z{z}.tif'  # change to match your own per-z-slice filenames ({z} is substituted with 0, 1, 2, ...); ignored if VOLUME_FILE is set
VOLUME_FILE = 'colon_dapi.tif'  # whole (Z,H,W) volume is this one combined multi-page tif instead of one file per z-slice (None = per-slice files)

# Intensity normalization range (1-99 percentile of a whole volume, computed on the first run and cached), one per
# volume: the range of the volume the reviewed views were cut from normalizes the training views, the range of
# DATA_DIR's volume normalizes everything predicted from it (new HITL views, 3D_segmentation). A missing
# TRAIN_NORM_CACHE is built from DATA_DIR's volume -- right only when the views were cut from it.
DATA_NORM_CACHE = os.path.join(DATA_DIR, 'global_norm.json')



#===============================================   human-in-the-loop   ===============================================
HITL_DIR = os.path.join(PROJECT_ROOT, 'data', 'R3-hitl-2')
MAIN_MODELS_DIR = os.path.join(HITL_DIR, 'models', 'main')
MAIN_CHECKPOINT_PATH = os.path.join(MAIN_MODELS_DIR, 'model.pth')
TRAIN_NORM_CACHE = os.path.join(HITL_DIR, 'global_norm.json')


# view, block, patch, model size setting

# cellpose-SAM segments objects of roughly cell size in fixed 256 x 256 px tiles. Objects much bigger than that segment
# poorly at the raw resolution, so images and labels are shrunk by 1 / RESCALE_FACTOR in xy before the model and the
# predicted masks are enlarged back: files on disk stay at the raw resolution. 1.0 = no rescale.
RESCALE_FACTOR = 1.0    # MUST be set for your data: compare a few factors on one region
REVIEW_SIZE_XY = 1024   # xy size (raw px) of one HITL review region -- set it to how much you want to correct at once (>= 256 * RESCALE_FACTOR)
REVIEW_SIZE_Z = 7       # z depth (slices) of one HITL review region; >= PATCH_SIZE_Z (when larger, training draws random z windows from it)
PATCH_SIZE_Z = 7        # z depth (slices) the model processes at once: training crops and inference windows; sets the U-Net's z pooling
NORM_Z_SUBSAMPLE = 1    # global intensity normalization (DATA_NORM_CACHE / TRAIN_NORM_CACHE): read every N-th slice for it.
                        # 1 = all slices; larger = faster on big volumes

AREA_RATIO = (REVIEW_SIZE_XY // int(256 * RESCALE_FACTOR)) ** 2   # model tiles (256 px after the downscale) per review region = training crops per region per epoch


# main model training
BOOTSTRAP_N_VIEWS = 1       # reviewed regions needed before the first training; until then new regions are predicted by the starting model as is
N_EPOCHS = 100              # epochs of every HITL round (each round trains from cellpose-SAM + a zero-init U-Net on all reviewed regions)
BATCH_SIZE = 1              # training crops per step; bound by GPU memory
LEARNING_RATE = 3e-3        # AdamW learning rate
WEIGHT_DECAY = 0.1          # AdamW weight decay
SCALE_RANGE = 0.5           # augmentation: each training crop is zoomed by a random factor in [1 - SCALE_RANGE/2, 1 + SCALE_RANGE/2]
EVAL_EVERY = 20             # AP on the training regions every this many epochs (a check only; there is no held-out set)
MIN_TRAIN_MASKS = 5         # a training crop holding fewer objects than this is redrawn (step5_train.py:random_rotate_and_resize)
UNFREEZE_LAST_N_BLOCKS = 0  # last ViT encoder blocks that are trained too; 0 = encoder frozen, only the 3D U-Net learns (GPU memory)

# loss term that keeps the predictions of adjacent slices alike, on a band along each object's outline: weight on the
# flow part, weight on the cell-probability part, and the band width (px at the model's resolution). Both weights 0 = off.
ZSMOOTH_BOUNDARY_WEIGHT_FLOW = 0.0
ZSMOOTH_BOUNDARY_WEIGHT_CELLPROB = 0.0
ZSMOOTH_BOUNDARY_DILATE_PX = 5
#=====================================================================================================================



#===============================================    post-processing    ===============================================
# Per-slice mask computation inside inference (step6 run_inference_cpsam3d -> compute_masks_2d), applied to the HITL
# regions and to 3D_segmentation alike. No z repair: linking the slices is left to the stitch.
FLOW_THRESHOLD = 0.4    # a piece whose flow disagrees with its own shape by more than this is deleted (cellpose's default 0.4)
MIN_SIZE_PX = 300       # pieces smaller than this (raw px) are deleted; keep it below the smallest real cross-section of your objects
#=====================================================================================================================




#===============================================    3D_segmentation    ===============================================
SEGMENTATION_DIR = os.path.join(DATA_DIR, 'segmentation')

# z-stitch (cellpose's stitch3D rule, stitch.py): adjacent-slice pieces are linked 1:1 when their IoU is at
# least this. Used by the stitch of the assembled per-slice masks.
STITCH_THRESHOLD = 0.25

# physical voxel size of the raw data -- MUST be set for your data.
XY_PIXEL_UM = 0.108     # xy pixel size (um)
Z_STEP_UM = 1.5         # z step between slices (um)
# z/xy ratio at the model's resolution. It shapes the U-Net's z pooling, so a checkpoint only fits data with the same
# ratio (it was built with this value).
Z_XY_ANISOTROPY = Z_STEP_UM / (XY_PIXEL_UM * RESCALE_FACTOR)

# Whole-volume inference runs block by block to fit in memory; the blocks are invisible to the model (it always works
# in 256 px tiles and PATCH_SIZE_Z windows). Each block is read with a margin on every side and only its centre is
# kept, so objects at a block border are segmented with their surroundings; objects cut by an xy border are then
# merged across it.
BLOCK_SIZE = 4096       # xy block size (raw px)
MARGIN = 128            # xy overlap (raw px) read around each block; about one object across or more
BLOCK_SIZE_Z = 64       # z block depth (slices)
MARGIN_Z = 16           # z overlap (slices) read above and below each block; more than one object's z extent
MIN_TOUCH_PIXELS = 3    # objects meeting across an xy block border are merged if they touch on at least this many px and are each other's best match (1-2 px contacts are noise)
#=====================================================================================================================

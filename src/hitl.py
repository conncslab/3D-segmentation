"""Human-in-the-loop main loop for the main segmentation model (CPSAM3D). Each run takes a
single coordinate the user picked directly in Neuroglancer etc. as a command-line argument:

    python hitl.py <x> <y> [z]

Every time it runs:
  1. Trains on everything the human has reviewed so far. A view counts as reviewed once
     <name>_masks_modified.tif exists (the human saves it after correcting). Every
     round RETRAINS FROM THE SAME STARTING POINT (cpsam_v2 + a zero-init U-Net) on ALL
     reviewed views at once, for N_EPOCHS -- never continues from a previous round's
     checkpoint. This follows
     Cellpose's own human-in-the-loop guidance (refs-code/cellpose-main/docs/train.rst,
     gui.rst): continuing from a prior retraining biases the model toward whichever images
     it already trained on, whereas starting fresh each time weighs every reviewed view
     equally. Only the ONE latest checkpoint is ever kept (MAIN_CHECKPOINT_PATH). Below
     BOOTSTRAP_N_VIEWS reviewed views nothing is trained yet at all.
     crops_per_epoch = accumulated view count * AREA_RATIO so the whole area gets sampled
     evenly.
  2. Extracts a new view at the grid cell the given coordinate falls in and predicts its
     per-slice masks -- with our own CPSAM3D once it has been trained, before that with the
     starting point as is (= plain cellpose-SAM, the zero-init U-Net adds nothing). Either way masks.tif is PLAIN PER-SLICE 2D
     segmentation, no z-stitching: training the main model only ever needs per-slice masks
     (labels_to_flows_3d renumbers ids independently per slice regardless), and a stitched
     preview would hide real per-slice problems.
  3. Prints which files to review. The human corrects the per-slice masks in any editor and
     saves them as <name>_masks_modified.tif next to the others; the next run trains on it.
     The model's own masks.tif is never overwritten, and a view without the modified file is
     silently skipped in training (list_existing_views).
"""
import os
import sys
import glob
import contextlib
import numpy as np
import torch
import tifffile
import cv2
from cellpose.transforms import resize_image

from config import (AREA_RATIO, PATCH_SIZE_Z, Z_XY_ANISOTROPY,
                     BOOTSTRAP_N_VIEWS, N_EPOCHS, BATCH_SIZE,
                     UNFREEZE_LAST_N_BLOCKS, LEARNING_RATE, WEIGHT_DECAY, SCALE_RANGE,
                     MIN_TRAIN_MASKS, EVAL_EVERY,
                     RESCALE_FACTOR, ZSMOOTH_BOUNDARY_WEIGHT_FLOW,
                     ZSMOOTH_BOUNDARY_WEIGHT_CELLPROB, ZSMOOTH_BOUNDARY_DILATE_PX,
                     MAIN_MODELS_DIR, MAIN_CHECKPOINT_PATH, HITL_DIR, TRAIN_NORM_CACHE, DATA_NORM_CACHE)
from step1_patch_prep import open_z_slice_memmaps, load_or_build_global_norm, read_view_at, view_name
from step2_data_input import get_label_files, imread
from step3_pre_processing import preprocess, pad_to_3channel, reshape_and_normalize_data
from step4_flow_vector import labels_to_flows_3d
from model.cpsam3d import CPSAM3D
from step5_train import train_model
from step6_inference_and_postprocessing import run_inference_cpsam3d, flow_stack_to_rgb


@contextlib.contextmanager
def tee_stdout(log_path):
    """Duplicates everything printed inside the block to both real stdout (so it's
    still visible if watching this run live) and log_path, verbatim."""
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    with open(log_path, 'w') as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            yield


def list_existing_views(hitl_dir):
    """Among <hitl_dir>'s view_*_img.tif files, returns only those that have a
    matching view_*_masks_modified.tif (the human's completed correction).

    A view with only _masks.tif (the model's raw prediction) and no
    _masks_modified.tif means either the human hasn't corrected it yet (normal
    waiting state) or a previous run died partway through -- silently treated as
    "doesn't exist yet"."""
    image_names = sorted(glob.glob(os.path.join(hitl_dir, 'view_*_img.tif')))
    complete = []
    for name in image_names:
        base = os.path.splitext(name)[0][:-len('_img')]
        if os.path.exists(base + '_masks_modified.tif'):
            complete.append(name)
    return complete


def parse_args(argv):
    """argv: sys.argv as-is. Returns (x, y, z). z is optional (defaults to None).
    Raises ValueError if arguments are invalid."""
    if len(argv) < 3:
        raise ValueError('Usage: python hitl.py <x> <y> [z]')
    try:
        x, y = int(argv[1]), int(argv[2])
    except ValueError:
        raise ValueError('x and y coordinates must be integers')
    z = None
    if len(argv) >= 4:
        try:
            z = int(argv[3])
        except ValueError:
            raise ValueError(f'Invalid z coordinate: {argv[3]} (must be integer)')
    return x, y, z


def rescale_training_data(images, labels, rescale_factor):
    """images/labels: lists of raw (Z,H,W) arrays (matching, same length). Downscales
    both in XY by 1/rescale_factor (images: linear, labels: nearest-neighbor to keep
    instance ids intact). No-op if rescale_factor == 1.0."""
    if rescale_factor == 1.0:
        return images, labels
    rsz = 1.0 / rescale_factor
    images_rsz = [resize_image(img, rsz=rsz, no_channels=True, interpolation=cv2.INTER_LINEAR) for img in images]
    labels_rsz = [resize_image(lbl, rsz=rsz, no_channels=True, interpolation=cv2.INTER_NEAREST) for lbl in labels]
    return images_rsz, labels_rsz


def init_model(device):
    """CPSAM3D at the starting point of every HITL round: cpsam_v2 + a zero-init U-Net."""
    # rdrop=0.0: CPSAM's default layer-drop only matters for an unfrozen encoder -- here
    # the encoder stays fully frozen, so rdrop would just perturb the residual base the
    # UNet has to correct by ~6-10% every step (measured) and create a train/eval
    # mismatch, for no regularization benefit (nothing trainable is being dropped).
    net = CPSAM3D(ps=8, nout=3, bsize=256, z_crop_size=PATCH_SIZE_Z,
                  z_xy_ratio=Z_XY_ANISOTROPY, rdrop=0.0).to(device)
    net.load_pretrained_cpsam(device=device)
    return net


def train_main_model(img_names, device, p_low, p_high):
    """A fresh CPSAM3D (init_model) fine-tuned on every reviewed view in img_names; saved to
    MAIN_CHECKPOINT_PATH and returned."""
    # torch.manual_seed before construction: with only a handful of corrected views,
    # held-out quality varies a lot by which random init the UNet's fresh layers happen
    # to draw (measured directly: three otherwise-identical training runs landed at
    # held-out AP@0.9 of 0.614/0.622/0.720 from seed alone).
    torch.manual_seed(1234)
    net = init_model(device)

    crops_per_epoch = len(img_names) * AREA_RATIO
    print(f'[hitl] {len(img_names)} reviewed views -- fine-tuning again from the starting point '
          f'(n_epochs={N_EPOCHS}, crops_per_epoch={crops_per_epoch}, eval_every={EVAL_EVERY}).')

    label_names, _ = get_label_files(img_names, mask_filter='_masks_modified', imf='_img')
    images = [imread(n) for n in img_names]
    labels = [imread(n) for n in label_names]
    images, labels = rescale_training_data(images, labels, RESCALE_FACTOR)
    images, labels, _, _, _ = preprocess(images, labels, None, None, channels=[2, 1],
                                          percentile_range=(p_low, p_high))
    images = pad_to_3channel(images)
    flows = labels_to_flows_3d(labels, files=img_names, use_gpu=True, device=device)

    with tee_stdout(os.path.join(MAIN_MODELS_DIR, 'train_log.txt')):
        train_model(
            net, device, flows, images,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            batch_size=BATCH_SIZE,
            scale_range=SCALE_RANGE,
            min_train_masks=MIN_TRAIN_MASKS,
            crops_per_epoch=crops_per_epoch,
            freeze_encoder=True,
            unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS,
            checkpoint_dir=None,
            save_every=N_EPOCHS,
            eval_every=EVAL_EVERY,
            patch_size_z=PATCH_SIZE_Z,
            zsmooth_boundary_weight_flow=ZSMOOTH_BOUNDARY_WEIGHT_FLOW,
            zsmooth_boundary_weight_cellprob=ZSMOOTH_BOUNDARY_WEIGHT_CELLPROB,
            zsmooth_boundary_dilate_px=ZSMOOTH_BOUNDARY_DILATE_PX,
        )
    net.save_model(MAIN_CHECKPOINT_PATH)
    print(f'[hitl] 본 모델 체크포인트 저장: {MAIN_CHECKPOINT_PATH}')
    return net


def predict_view(net, view, device, p_low, p_high, name):
    """Our trained CPSAM3D on one view: <name>_masks.tif (per-slice masks) and
    <name>_flow_vis.tif (flow preview) into HITL_DIR."""
    data, _, _ = reshape_and_normalize_data([view.astype(np.float32)], test_data=None,
                                            channels=[2, 1], percentile_range=(p_low, p_high))
    data = pad_to_3channel(data)
    masks2d, flows = run_inference_cpsam3d(net, data, device, rescale_factor=RESCALE_FACTOR,
                                           z_bsize=PATCH_SIZE_Z, return_flow=True)
    masks = masks2d[0].astype(np.uint16)
    tifffile.imwrite(os.path.join(HITL_DIR, f'{name}_masks.tif'), masks)
    tifffile.imwrite(os.path.join(HITL_DIR, f'{name}_flow_vis.tif'), flow_stack_to_rgb(flows[0][0], flows[0][1]))


def main():
    try:
        x, y, z = parse_args(sys.argv)
    except ValueError as e:
        print(e)
        sys.exit(1)
    if not torch.cuda.is_available():
        raise ValueError('no GPU')
    device = torch.device('cuda')

    os.makedirs(HITL_DIR, exist_ok=True)
    os.makedirs(MAIN_MODELS_DIR, exist_ok=True)
    mmaps = open_z_slice_memmaps()
    # training views are normalized with the range of the volume they were cut from, the new view
    # (cut from DATA_DIR's volume) with that volume's own range
    data_p_low, data_p_high = load_or_build_global_norm(mmaps, DATA_NORM_CACHE)

    reviewed = list_existing_views(HITL_DIR)
    if len(reviewed) < BOOTSTRAP_N_VIEWS:
        print(f'[hitl] bootstrap: {len(reviewed)}/{BOOTSTRAP_N_VIEWS} reviewed views -- nothing trained yet, '
              f'the next view is predicted with the starting point (plain cellpose-SAM) as is.')
        net = init_model(device)
    else:
        train_p_low, train_p_high = load_or_build_global_norm(mmaps, TRAIN_NORM_CACHE)
        net = train_main_model(reviewed, device, train_p_low, train_p_high)

    # ---- extract the next view at the given coordinate and predict its per-slice masks ----
    view, row, col, z0 = read_view_at(mmaps, x, y, z=z)
    name = view_name(row, col, z_start=z0 if z is not None else None)
    img_path = os.path.join(HITL_DIR, f'{name}_img.tif')
    tifffile.imwrite(img_path, view)
    net.eval()
    predict_view(net, view, device, data_p_low, data_p_high, name)

    coord_str = f'({x},{y},{z})' if z is not None else f'({x},{y})'
    z_str = f', z0={z0}' if z is not None else ''
    print(f'[hitl] 새 view 추출/추론 완료: {name} (좌표={coord_str} -> grid row={row}, col={col}{z_str})')
    print(f'  -> 검토할 파일: 원본 {os.path.abspath(img_path)}')
    print(f'                  모델 마스크 {os.path.abspath(os.path.join(HITL_DIR, f"{name}_masks.tif"))}')
    print(f'  -> 고친 per-slice 마스크를 {name}_masks_modified.tif 이름으로 같은 폴더에 저장하면 다음 실행 때 학습에 '
          f'들어갑니다 (없으면 이 view는 학습에서 빠집니다).')
    n_after_this = len(reviewed) + 1
    if n_after_this < BOOTSTRAP_N_VIEWS:
        print(f'  -> 아직 bootstrap: 검토 후 {n_after_this}/{BOOTSTRAP_N_VIEWS}개, '
              f'{BOOTSTRAP_N_VIEWS - n_after_this}개 더 모아야 첫 학습이 시작됩니다.')


if __name__ == '__main__':
    main()

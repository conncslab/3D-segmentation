"""Step1 -- mosaic access and view extraction for HITL.

Your dataset (one uint16 tif file per z-slice, each potentially many GB) can't be
loaded into memory all at once, so each z-slice file is opened separately as a
tifffile.memmap and only the needed region is read as a window (the whole thing is
never loaded at once).

## Global normalization range
Computing a percentile within just one view (1024x1024) makes even cells whose
signal would be negligible relative to the whole dataset look relatively bright
inside that one view, turning them into segmentation targets. So the whole mosaic
(all z's) is scanned once, streaming, to build a pixel-value histogram, from which the
1~99 percentile range is derived and applied identically to every view. Pixels
whose value is exactly 0 (unscanned or masked regions) are NOT excluded separately --
included as-is, aiming for the same effect as normalizing against the whole image in
Neuroglancer (near-noise, mostly peripheral faint cells become invisible). The scan
result is cached to a file and reused -- one cache per volume (config.DATA_NORM_CACHE for
the volume being predicted, config.TRAIN_NORM_CACHE for the one the training views come from).

## View selection
No automatic tissue-priority logic -- takes the (x, y) coordinate the user picked
directly in another visualization tool (Neuroglancer) as a command-line argument. The
entire 1024x1024 grid cell that coordinate falls in is used as the view (there's no
queue tracking which grid cells have already been used -- entirely left to the
user's judgment).
"""
import os
import json
import numpy as np
import tifffile

from config import (PROJECT_NAME, DATA_DIR, Z_FILE_PATTERN, VOLUME_FILE, REVIEW_SIZE_XY,
                    REVIEW_SIZE_Z, NORM_Z_SUBSAMPLE)

BAND_ROWS = 4096  # rows read at once during the histogram scan (~460MB/z/band, never loads the whole thing at once)


def open_z_slice_memmaps(data_dir=DATA_DIR, volume_file=VOLUME_FILE):
    """Opens the mosaic's z-slices as something indexable by [z] -> (H,W), with len()
    giving the total z count -- every other function in this file (and 3D_segmentation.py) only ever
    does mmaps[z] or len(mmaps), so both layouts below satisfy that contract identically.
    The z count is the data's own: every slice of the volume file, or every per-slice file.

    volume_file set (e.g. kidney): the whole (Z,H,W) volume is one combined multi-page
    tif -- tifffile.memmap already returns a real 3D memmap for this, so it's returned as-is.

    volume_file=None (default, e.g. R3): opens the per-z-slice files (Z_FILE_PATTERN, z = 0, 1, 2, ...
    up to the first missing one) as a list of 2D memmaps, in order from z0 (bottom) to the last z."""
    if volume_file is not None:
        mm = tifffile.memmap(os.path.join(data_dir, volume_file))
        if mm.ndim != 3:
            raise ValueError(f'Expected a (Z,H,W) volume in {volume_file}, got shape {mm.shape}')
        return mm

    n_z = 0
    while os.path.exists(os.path.join(data_dir, Z_FILE_PATTERN.format(z=n_z))):
        n_z += 1
    if n_z == 0:
        raise FileNotFoundError(f'No {Z_FILE_PATTERN.format(z=0)} in {data_dir}')
    mmaps = [tifffile.memmap(os.path.join(data_dir, Z_FILE_PATTERN.format(z=z))) for z in range(n_z)]
    shapes = {mm.shape for mm in mmaps}
    dtypes = {mm.dtype for mm in mmaps}
    if len(shapes) > 1 or len(dtypes) > 1:
        raise ValueError(f'Mismatched shape/dtype across z-slice files: shapes={shapes}, dtypes={dtypes}')
    return mmaps


def get_mosaic_shape(mmaps):
    return mmaps[0].shape  # (H, W)


def compute_global_percentile(mmaps, low=1, high=99, band_rows=BAND_ROWS, z_subsample=NORM_Z_SUBSAMPLE):
    """Streams and accumulates a pixel-value histogram over the mosaic to derive the
    (low, high) percentile values. Pixels valued 0 are included as-is.
    If z_subsample > 1, slices are sampled at step z_subsample for faster scanning over large Z volumes."""
    H, W = get_mosaic_shape(mmaps)
    hist = np.zeros(65536, dtype=np.int64)
    step = max(1, int(z_subsample))
    slice_indices = list(range(0, len(mmaps), step))
    for zi in slice_indices:
        sub_str = f' (subsampled step={step})' if step > 1 else ''
        print(f'[step1] Accumulating histogram for global percentile: z{zi}{sub_str}...')
        mm = mmaps[zi]
        for row_start in range(0, H, band_rows):
            row_end = min(row_start + band_rows, H)
            band = mm[row_start:row_end, :]
            h, _ = np.histogram(band, bins=65536, range=(0, 65536))
            hist += h
    cdf = np.cumsum(hist).astype(np.float64)
    total = cdf[-1]
    p_low = float(np.searchsorted(cdf, total * low / 100.0))
    p_high = float(np.searchsorted(cdf, total * high / 100.0))
    return p_low, p_high


def load_or_build_global_norm(mmaps, cache_path, low=1, high=99, z_subsample=NORM_Z_SUBSAMPLE):
    """(p_low, p_high) of the volume behind cache_path: read from the cache, or scanned from mmaps and cached."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            d = json.load(f)
        return d['p_low'], d['p_high']
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print('[step1] Global normalization cache not found -- scanning mosaic (one-time, cached thereafter)...')
    p_low, p_high = compute_global_percentile(mmaps, low=low, high=high, z_subsample=z_subsample)
    with open(cache_path, 'w') as f:
        json.dump({'p_low': p_low, 'p_high': p_high, 'low_pct': low, 'high_pct': high, 'z_subsample': z_subsample}, f, indent=2)
    print(f'[step1] Global normalization range computed: p{low}={p_low:.1f}, p{high}={p_high:.1f}')
    return p_low, p_high


def grid_cell_for_coord(x, y, view_size=REVIEW_SIZE_XY):
    return y // view_size, x // view_size  # (row, col)


def grid_slice_for_z(z, view_size_z=REVIEW_SIZE_Z):
    """Returns the grid slice index and starting z0 for coordinate z."""
    if z is None:
        return 0, 0
    z_idx = z // view_size_z
    return z_idx, z_idx * view_size_z


def view_name(row, col, z_start=None):
    if z_start is not None:
        return f'view_z{z_start:03d}_r{row:03d}_c{col:03d}'
    return f'view_r{row:03d}_c{col:03d}'


def read_block(mmaps, y0, x0, h, w, z0=0, depth=None):
    """Reads an h x w region (doesn't need to be square) with (y0,x0) as its top-left
    corner, across depth z-slices starting at z0.
    Any part that goes past the canvas boundary is filled with 0 (needed for 3D_segmentation.py's edge blocks).
    If depth is None, reads all slices from z0 to the end of mmaps.
    The return value is a real in-memory array, not a memmap, so it can be saved as-is."""
    H, W = get_mosaic_shape(mmaps)
    total_z = len(mmaps)
    if depth is None:
        depth = total_z - z0

    y1, x1 = min(y0 + h, H), min(x0 + w, W)
    copy_h, copy_w = max(0, y1 - y0), max(0, x1 - x0)
    block = np.zeros((depth, h, w), dtype=mmaps[0].dtype)

    z_end = min(z0 + depth, total_z)
    for out_z, z in enumerate(range(z0, z_end)):
        block[out_z, :copy_h, :copy_w] = mmaps[z][y0:y1, x0:x1]
    return block


def read_view_at(mmaps, x, y, z=None, view_size=REVIEW_SIZE_XY, view_size_z=REVIEW_SIZE_Z):
    """Reads the view_size x view_size region of the grid cell that the (x, y)
    coordinate falls in, across view_size_z slices starting around z, from memmaps,
    combined into (view_size_z, view_size, view_size).
    The return value is a real array in memory, not a memmap, so it can be saved
    directly with tifffile.imwrite (becomes a real file any viewer can open)."""
    row, col = grid_cell_for_coord(x, y, view_size)
    y0, x0 = row * view_size, col * view_size
    total_z = len(mmaps)
    depth = view_size_z if view_size_z is not None else total_z
    depth = min(depth, total_z)

    if z is not None:
        _, z0 = grid_slice_for_z(z, depth)
        if z0 + depth > total_z:
            z0 = max(0, total_z - depth)
    else:
        z0 = 0

    block = read_block(mmaps, y0, x0, view_size, view_size, z0=z0, depth=depth)
    return block, row, col, z0

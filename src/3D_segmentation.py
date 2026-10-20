"""Final exhaustive inference over the whole raw volume, followed by assembling the resulting
blocks into one combined PER-SLICE (2D) segmentation.

This step deliberately does NOT produce a 3D-stitched result; the one and only 3D linking
happens afterward over the fully assembled per-slice array (the stitch: cellpose's stitch3D
rule, stitch.py).

Order of the whole flow this script sits in:
  1. the HITL loop (hitl.py) trains the main model
  2. THIS SCRIPT: whole-volume per-slice 2D masks, assembled into one array
  3. (optional) a human edits the per-slice masks
  4. one 3D linking over the (possibly edited) per-slice masks (the stitch)

    python 3D_segmentation.py

## Block inference

Uses whatever the HITL loop (hitl.py) has produced so far (MAIN_CHECKPOINT_PATH -- every
round retrains from scratch on all reviewed views, so there's only ever this one file).

Processes the whole canvas sequentially in large blocks (default 4096x4096xBLOCK_SIZE_Z),
chunked along z the same way as xy (BLOCK_SIZE_Z/MARGIN_Z, config.py) -- purely a memory-
management split for the neural net's own preprocessing buffer; run_inference_cpsam3d
further chunks the block interior into PATCH_SIZE_Z-thick z windows and 256x256 SAM xy
tiles to match what the model was actually trained on regardless of block size. Same as
patch extraction, normalization uses the volume's global (p_low, p_high) as-is
(step1_patch_prep.load_or_build_global_norm on config.DATA_NORM_CACHE, loaded instantly if
already cached).

Each block saves its own per-slice masks, margin-cropped to the block's pure
(non-overlapping) region. z-stitching only ever happens globally, over the fully assembled
array (never per z-chunk with a seam to reconcile afterward), so the single 3D linking sees
no artificial block boundaries.

Completed blocks are recorded in SEGMENTATION_DIR/progress.json, so re-running after an
interruption resumes where it left off.

## Merge + combine

This step assembles every block's saved PRE-STITCH per-slice masks into one global
per-slice array. Z-neighbors need no merge at all here (nothing is z-stitched in this
script; see the module docstring), but XY-neighbor blocks (when BLOCK_SIZE doesn't cover
the full width/height in one block) still do: they need the touching-face union-find merge
(accumulate_touches), and it has to run PER Z-SLICE, not
aggregated across a block's whole z-depth. pred_masks2d labels are per-slice-independent
(compute_masks_2d renumbers every slice from scratch), so a touching-pixel count summed
across multiple z-slices would conflate unrelated objects that merely share a coincidental
label value at different z.

Concretely:
1. For every z-chunk (zrow), for every z-slice inside it, gather that slice's array from
   every (row,col) block present, union xy-adjacent blocks' labels via
   accumulate_touches() (fresh UnionFind per slice), and assign each resulting group a
   freshly incrementing GLOBAL id -- written directly into one big in-memory
   (total_Z,total_H,total_W) uint32 array (assemble_predmasks2d_combined()). When the
   volume fits in one xy block, this union-find never finds a neighbor and degenerates to
   a plain per-block relabel, which is the correct behavior.
2. That assembled array is the deliverable, saved as PREDMASKS2D_PATH -- exactly what
   the 3D linking (the stitch) needs, with no coordinate translation, since it's already
   in one global coordinate space.

Known scaling limit (not solved here): assemble_predmasks2d_combined() builds the WHOLE
per-slice array as one dense in-RAM uint32 ndarray, Z x H x W x 4 bytes. A volume whose
per-slice mask doesn't fit in RAM that way needs it written to a memmap instead.

The global id count can exceed uint16's range (65535), so results are uint32.
"""
import os
import json
import numpy as np
import torch
import tifffile
import fastremap

from config import (PROJECT_NAME, MAIN_CHECKPOINT_PATH, SEGMENTATION_DIR, DATA_NORM_CACHE,
                     PATCH_SIZE_Z, Z_XY_ANISOTROPY, RESCALE_FACTOR,
                     BLOCK_SIZE, MARGIN, BLOCK_SIZE_Z, MARGIN_Z, MIN_TOUCH_PIXELS)
from step1_patch_prep import open_z_slice_memmaps, read_block, get_mosaic_shape, load_or_build_global_norm
from step3_pre_processing import pad_to_3channel, reshape_and_normalize_data
from model.cpsam3d import CPSAM3D
from step6_inference_and_postprocessing import run_inference_cpsam3d
from utils.utils import UnionFind

OUT_DIR = os.environ.get('CPSAM3D_OUT_DIR_OVERRIDE', SEGMENTATION_DIR)

PROGRESS_FILENAME = 'progress.json'

PREDMASKS2D_PATH = os.path.join(OUT_DIR, PROJECT_NAME + '_predmasks2d_combined.tif')


def latest_checkpoint():
    """The main model checkpoint the HITL loop (hitl.py) has produced.
    Every HITL round retrains from scratch on all reviewed views (Cellpose's own
    human-in-the-loop guidance -- see hitl.py), so there is only ever the ONE most
    recent checkpoint (MAIN_CHECKPOINT_PATH), never a round_NNNN history. Raises a clear
    error if it hasn't been trained yet.

    CPSAM3D_CKPT_OVERRIDE env var points this at any other CPSAM3D checkpoint instead."""
    override = os.environ.get('CPSAM3D_CKPT_OVERRIDE')
    if override:
        return override
    if os.path.exists(MAIN_CHECKPOINT_PATH):
        return MAIN_CHECKPOINT_PATH
    raise RuntimeError(f'{MAIN_CHECKPOINT_PATH} not found -- '
                        'at least one HITL round (hitl.py) must finish before running 3D_segmentation.')


def progress_path():
    return os.path.join(OUT_DIR, PROGRESS_FILENAME)


def load_or_init_progress(grid_shape):
    path = progress_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'grid_shape': list(grid_shape), 'block_size': BLOCK_SIZE, 'margin': MARGIN,
            'block_size_z': BLOCK_SIZE_Z, 'margin_z': MARGIN_Z, 'completed': []}


def save_progress(progress):
    path = progress_path()
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, path)


def run_block_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise ValueError('no GPU')

    os.makedirs(OUT_DIR, exist_ok=True)

    mmaps = open_z_slice_memmaps()
    H, W = get_mosaic_shape(mmaps)
    total_z = len(mmaps)
    p_low, p_high = load_or_build_global_norm(mmaps, DATA_NORM_CACHE)

    n_zblocks = int(np.ceil(total_z / BLOCK_SIZE_Z))
    n_rows = int(np.ceil(H / BLOCK_SIZE))
    n_cols = int(np.ceil(W / BLOCK_SIZE))

    progress = load_or_init_progress((n_zblocks, n_rows, n_cols))
    completed = {tuple(rc) for rc in progress['completed']}

    ckpt = latest_checkpoint()
    print(f'[3D_segmentation] checkpoint: {ckpt}')
    net = CPSAM3D(ps=8, nout=3, bsize=256, z_crop_size=PATCH_SIZE_Z,
                  z_xy_ratio=Z_XY_ANISOTROPY).to(device)
    net.load_model(ckpt, device=device)
    net.eval()

    total = n_zblocks * n_rows * n_cols
    print(f'[3D_segmentation] {n_zblocks}x{n_rows}x{n_cols}={total} blocks in total, {len(completed)} already done')

    for zrow in range(n_zblocks):
        z0, z1 = zrow * BLOCK_SIZE_Z, min(total_z, (zrow + 1) * BLOCK_SIZE_Z)
        zm0, zm1 = max(0, z0 - MARGIN_Z), min(total_z, z1 + MARGIN_Z)

        for row in range(n_rows):
            for col in range(n_cols):
                if (zrow, row, col) in completed:
                    continue

                y0, x0 = row * BLOCK_SIZE, col * BLOCK_SIZE
                y1, x1 = min(H, y0 + BLOCK_SIZE), min(W, x0 + BLOCK_SIZE)
                ym0, xm0 = max(0, y0 - MARGIN), max(0, x0 - MARGIN)
                ym1, xm1 = min(H, y1 + MARGIN), min(W, x1 + MARGIN)

                block = read_block(mmaps, ym0, xm0, ym1 - ym0, xm1 - xm0, z0=zm0, depth=zm1 - zm0)

                data, _, _ = reshape_and_normalize_data([block.astype(np.float32)], test_data=None,
                                                           channels=[2, 1], percentile_range=(p_low, p_high))
                data = pad_to_3channel(data)

                mask2d = run_inference_cpsam3d(net, data, device, rescale_factor=RESCALE_FACTOR,
                                               z_bsize=PATCH_SIZE_Z)[0]

                # crop off the margin (xy AND z), keeping only the pure block region
                cy0, cx0 = y0 - ym0, x0 - xm0
                cy1, cx1 = cy0 + (y1 - y0), cx0 + (x1 - x0)
                cz0 = z0 - zm0
                cz1 = cz0 + (z1 - z0)
                mask2d_center = mask2d[cz0:cz1, cy0:cy1, cx0:cx1]

                tifffile.imwrite(predmasks2d_block_path(zrow, row, col), mask2d_center)

                progress['completed'].append([zrow, row, col])
                save_progress(progress)

                done = len(progress['completed'])
                if done % 10 == 0 or done == total:
                    print(f'[3D_segmentation] {done}/{total} blocks done (block {zrow},{row},{col})')

    print('[3D_segmentation] block inference over the whole volume done.')


def predmasks2d_block_path(zrow, row, col):
    return os.path.join(OUT_DIR, f'block_{zrow:04d}_{row:04d}_{col:04d}_predmasks2d.tif')


def accumulate_touches(uf, key_a, edge_a, key_b, edge_b):
    """edge_a, edge_b: label arrays of matching rank (either (Z,L) for an xy-neighbor
    face, or (H,W) for a z-neighbor face -- both cropped to their common shape first).

    A label pair unions only if it clears MIN_TOUCH_PIXELS AND each is the OTHER's single
    strongest touching partner across this face (mutual best match) -- plain "union on any
    touch >= MIN_TOUCH_PIXELS" is transitive (A-B touch + B-C touch unions A,B,C together
    even if A and C are unrelated), and with a small MIN_TOUCH_PIXELS against a full block
    face (thousands of instances, jagged per-slice mask boundaries) such chains can fuse a
    large share of the foreground into one global id. The mutual-best-match requirement
    keeps each union one-to-one."""
    shape = tuple(min(sa, sb) for sa, sb in zip(edge_a.shape, edge_b.shape))
    slc = tuple(slice(0, s) for s in shape)
    a = edge_a[slc].astype(np.int64)
    b = edge_b[slc].astype(np.int64)
    both = (a > 0) & (b > 0)
    if not np.any(both):
        return
    pairs = np.stack([a[both], b[both]], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)

    best_b_for_a, best_a_for_b = {}, {}
    for (la, lb), cnt in zip(uniq, counts):
        la, lb, cnt = int(la), int(lb), int(cnt)
        if cnt < MIN_TOUCH_PIXELS:
            continue
        if la not in best_b_for_a or cnt > best_b_for_a[la][1]:
            best_b_for_a[la] = (lb, cnt)
        if lb not in best_a_for_b or cnt > best_a_for_b[lb][1]:
            best_a_for_b[lb] = (la, cnt)

    for la, (lb, cnt) in best_b_for_a.items():
        if best_a_for_b.get(lb, (None, None))[0] == la:
            uf.union((*key_a, la), (*key_b, lb))


def assemble_predmasks2d_combined(progress):
    """Builds the ONE global (total_Z,total_H,total_W) uint32 pre-stitch per-slice array
    out of every block's own saved predmasks2d file -- see module docstring for why the
    xy-merge has to run per z-slice rather than aggregated across a block's z-depth."""
    block_size = progress['block_size']
    block_size_z = progress['block_size_z']
    n_zblocks, n_rows, n_cols = progress['grid_shape']
    completed_set = {tuple(rc) for rc in progress['completed']}

    # each zrow's block(s) share one z-depth (found by peeking at whichever block in that
    # zrow is present) and together tile the full xy extent -- use those to size the output.
    # tifffile.imread's own returned shape is used (not TiffFile page/sample metadata,
    # which is ambiguous for a small leading axis) so this exactly matches whatever
    # ndarray shape was originally passed to imwrite.
    block_shapes = {}
    for zrow, row, col in completed_set:
        if (zrow, row, col) not in block_shapes:
            block_shapes[(zrow, row, col)] = tifffile.imread(predmasks2d_block_path(zrow, row, col)).shape

    total_z = max(zrow * block_size_z + block_shapes[(zrow, row, col)][0]
                 for (zrow, row, col) in block_shapes)
    total_y = max(row * block_size + block_shapes[(zrow, row, col)][1]
                 for (zrow, row, col) in block_shapes)
    total_x = max(col * block_size + block_shapes[(zrow, row, col)][2]
                 for (zrow, row, col) in block_shapes)
    print(f'[3D_segmentation] per-slice combined shape: '
          f'({total_z}, {total_y}, {total_x}) uint32 (~{total_z * total_y * total_x * 4 / 1e9:.1f}GB)')

    combined = np.zeros((total_z, total_y, total_x), dtype=np.uint32)
    next_gid = [1]

    for zrow in range(n_zblocks):
        present = [(row, col) for row in range(n_rows) for col in range(n_cols)
                  if (zrow, row, col) in completed_set]
        if not present:
            continue
        blocks = {(row, col): tifffile.imread(predmasks2d_block_path(zrow, row, col))
                 for row, col in present}
        chunk_depth = block_shapes[(zrow, *present[0])][0]
        z0 = zrow * block_size_z

        for local_z in range(chunk_depth):
            slices = {rc: arr[local_z] for rc, arr in blocks.items()}

            # xy-merge, this slice only -- a fresh UnionFind so nothing leaks across z
            uf = UnionFind()
            for (row, col), arr in slices.items():
                if (row, col + 1) in slices:
                    accumulate_touches(uf, (row, col), arr[:, -1],
                                       (row, col + 1), slices[(row, col + 1)][:, 0])
                if (row + 1, col) in slices:
                    accumulate_touches(uf, (row, col), arr[-1, :],
                                       (row + 1, col), slices[(row + 1, col)][0, :])

            root_to_gid = {}
            for (row, col), arr in slices.items():
                ids = fastremap.unique(arr)
                ids = ids[ids != 0]
                if ids.size == 0:
                    continue
                lut = np.zeros(int(ids.max()) + 1, dtype=np.uint32)
                for lid in ids:
                    root = uf.find((row, col, int(lid)))
                    if root not in root_to_gid:
                        root_to_gid[root] = next_gid[0]
                        next_gid[0] += 1
                    lut[lid] = root_to_gid[root]
                y0, x0 = row * block_size, col * block_size
                h, w = arr.shape
                combined[z0 + local_z, y0:y0 + h, x0:x0 + w] = lut[arr]

        print(f'[3D_segmentation] z-chunk {zrow + 1}/{n_zblocks} assembled '
              f'({len(present)} blocks, depth={chunk_depth})')

    return combined


def run_merge_and_combine():
    with open(progress_path()) as f:
        progress = json.load(f)

    combined = assemble_predmasks2d_combined(progress)
    tifffile.imwrite(PREDMASKS2D_PATH, combined, bigtiff=True)
    print(f'[3D_segmentation] per-slice (2D) masks saved: {PREDMASKS2D_PATH} '
          f'(edit them if needed, then link them in 3D with stitch.py)')


def main():
    run_block_inference()
    run_merge_and_combine()


if __name__ == '__main__':
    main()

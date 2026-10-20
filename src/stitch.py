"""Links the per-slice (2D) masks of a volume into 3D objects with cellpose's own stitch3D rule
(cellpose.utils.stitch3D): between adjacent slices, a z+1 piece joins the z piece it overlaps best
when their IoU >= STITCH_THRESHOLD and it is also that z piece's best z+1 partner -- 1:1, so a chain
of links is one 3D object.

    python stitch.py <masks2d.tif> [<out.tif>] [--threshold T]

masks2d.tif: (Z,H,W) per-slice masks -- 3D_segmentation.py's <PROJECT_NAME>_predmasks2d_combined.tif,
a hitl view's masks, ... Ids may be local to each slice or unique over the volume: a piece is (z, id).
out.tif: (Z,H,W) uint32 3D labels 1..N, numbered in order of first appearance (z, then input id).
Default: <masks2d>_stitched.tif next to the input.

Streams, so a volume much larger than RAM fits: slices are read
BAND_ROWS rows at a time from a memmap of the input, overlaps are counted per overlapping id pair
(never a dense id x id IoU matrix like cellpose's), and the labels go straight into an on-disk
memmap of the output.

One deliberate difference from cellpose.utils.stitch3D: there, while nothing has been stitched yet,
a slice that follows an empty slice keeps its own ids, which can repeat ids of earlier slices and
fuse unrelated objects; here every piece is its own node, so that can't happen.
"""
import os
import argparse
import numpy as np
import tifffile
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from config import STITCH_THRESHOLD

BAND_ROWS = 2048   # rows of a slice handled at once (BAND_ROWS x width px per band)


def open_masks(path):
    """(Z,H,W) per-slice masks: a read-only memmap when the tif allows one (uncompressed,
    contiguous -- e.g. what 3D_segmentation.py writes), else read into memory."""
    try:
        masks = tifffile.memmap(path, mode='r')
    except ValueError:
        masks = tifffile.imread(path)
    if masks.ndim != 3:
        raise ValueError(f'expected (Z,H,W) per-slice masks in {path}, got shape {masks.shape}')
    return masks


def slice_areas(masks, z):
    """Pixel count of every id on slice z (index = id; index 0, the background, set to 0)."""
    area = np.zeros(1, np.int64)
    for y0 in range(0, masks.shape[1], BAND_ROWS):
        c = np.bincount(np.asarray(masks[z, y0:y0 + BAND_ROWS]).ravel())
        if c.size > area.size:
            c[:area.size] += area
            area = c
        else:
            area[:c.size] += c
    area[0] = 0
    return area


def slice_overlaps(masks, z):
    """Every pair of pieces overlapping between slices z+1 and z: (id on z+1, id on z, shared px)."""
    keys, counts = [], []
    for y0 in range(0, masks.shape[1], BAND_ROWS):
        a = np.asarray(masks[z + 1, y0:y0 + BAND_ROWS])
        b = np.asarray(masks[z, y0:y0 + BAND_ROWS])
        both = (a > 0) & (b > 0)
        k, c = np.unique((a[both].astype(np.uint64) << np.uint64(32)) | b[both].astype(np.uint64),
                         return_counts=True)
        keys.append(k)
        counts.append(c)
    keys, inv = np.unique(np.concatenate(keys), return_inverse=True)
    counts = np.bincount(inv, weights=np.concatenate(counts)).astype(np.int64)
    return (keys >> np.uint64(32)).astype(np.int64), (keys & np.uint64(0xffffffff)).astype(np.int64), counts


def link_pair(id_z1, id_z, shared, area_z1, area_z, threshold):
    """cellpose's stitch3D rule on one slice pair, over the overlapping pairs only. Returns the linked
    (ids on z+1, ids on z), at most one z piece per z+1 piece."""
    iou = shared / (area_z1[id_z1] + area_z[id_z] - shared)
    keep = iou >= threshold                                  # iou[iou < stitch_threshold] = 0
    id_z1, id_z, iou = id_z1[keep], id_z[keep], iou[keep]
    best_of_z = np.zeros(area_z.size)
    np.maximum.at(best_of_z, id_z, iou)
    keep = iou >= best_of_z[id_z]                            # iou[iou < iou.max(axis=0)] = 0
    id_z1, id_z, iou = id_z1[keep], id_z[keep], iou[keep]
    if id_z1.size == 0:
        return id_z1, id_z
    order = np.lexsort((id_z, -iou, id_z1))                  # iou.argmax(axis=1); a tie goes to the smaller id
    id_z1, id_z = id_z1[order], id_z[order]
    first = np.r_[True, id_z1[1:] != id_z1[:-1]]
    return id_z1[first], id_z[first]


def stitch(masks, out, threshold=STITCH_THRESHOLD):
    """masks: (Z,H,W) per-slice masks (array or memmap); out: (Z,H,W) uint32 array or memmap the 3D labels
    are written into. Returns the number of 3D objects."""
    Z, H = masks.shape[:2]

    # nodes: every (z, id) piece, numbered slice by slice in id order
    areas, ids, offsets, n_nodes = [], [], [], 0
    for z in range(Z):
        area = slice_areas(masks, z)
        areas.append(area)
        ids.append(np.nonzero(area)[0])
        offsets.append(n_nodes)
        n_nodes += ids[-1].size
    node_of = []
    for z in range(Z):
        lut = np.full(areas[z].size, -1, np.int64)
        lut[ids[z]] = offsets[z] + np.arange(ids[z].size)
        node_of.append(lut)

    # links between adjacent slices
    src, dst = [], []
    for z in range(Z - 1):
        id_z1, id_z = link_pair(*slice_overlaps(masks, z), areas[z + 1], areas[z], threshold)
        src.append(node_of[z + 1][id_z1])
        dst.append(node_of[z][id_z])
        print(f'[stitch] z{z}-z{z + 1}: {id_z1.size} links ({ids[z].size} / {ids[z + 1].size} pieces)')
    src = np.concatenate(src) if src else np.zeros(0, np.int64)
    dst = np.concatenate(dst) if dst else np.zeros(0, np.int64)

    # one 3D object per chain of links, numbered by first appearance
    graph = coo_matrix((np.ones(src.size, np.int8), (src, dst)), shape=(n_nodes, n_nodes))
    n_obj, comp = connected_components(graph, directed=False)
    _, first_node = np.unique(comp, return_index=True)
    rank = np.empty(n_obj, np.int64)
    rank[np.argsort(first_node)] = np.arange(n_obj)
    gid = (rank[comp] + 1).astype(np.uint32)

    for z in range(Z):
        lut = np.zeros(areas[z].size, np.uint32)
        lut[ids[z]] = gid[offsets[z]:offsets[z] + ids[z].size]
        for y0 in range(0, H, BAND_ROWS):
            out[z, y0:y0 + BAND_ROWS] = lut[np.asarray(masks[z, y0:y0 + BAND_ROWS])]

    z_extent = np.bincount(np.bincount(comp), minlength=Z + 1)[1:]
    print(f'[stitch] {n_nodes} pieces -> {n_obj} 3D objects; objects by number of pieces: '
          + ', '.join(f'{k + 1}: {c}' for k, c in enumerate(z_extent) if c))
    return n_obj


def stitch_volume(masks, threshold=STITCH_THRESHOLD):
    """In-memory variant for small volumes (e.g. one view): returns the (Z,H,W) uint32 3D labels."""
    out = np.zeros(masks.shape, np.uint32)
    stitch(masks, out, threshold)
    return out


def main():
    ap = argparse.ArgumentParser(description='Stitch per-slice masks into 3D objects (cellpose stitch3D rule).')
    ap.add_argument('masks2d', help='(Z,H,W) per-slice masks tif')
    ap.add_argument('out', nargs='?', help='output tif (default: <masks2d>_stitched.tif)')
    ap.add_argument('--threshold', type=float, default=STITCH_THRESHOLD, help=f'IoU threshold (default {STITCH_THRESHOLD})')
    args = ap.parse_args()
    out_path = args.out or os.path.splitext(args.masks2d)[0] + '_stitched.tif'
    if os.path.abspath(out_path) == os.path.abspath(args.masks2d):
        raise ValueError('output path must differ from the input')

    masks = open_masks(args.masks2d)
    out = tifffile.memmap(out_path, shape=masks.shape, dtype=np.uint32, bigtiff=True)
    stitch(masks, out, args.threshold)
    out.flush()
    del out
    print(f'[stitch] saved: {out_path}')


if __name__ == '__main__':
    main()

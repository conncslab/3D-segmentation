#========================================================================================
# Inference post-processing (GPU only). There is no dZ, so flow tracing is
# always per-z-slice, and compute_masks_2d turns each slice's traced positions into
# instances whose ids mean nothing across z. Linking them into 3D cells is a separate pass
# over the assembled per-slice volume (stitch.py: cellpose's own stitch3D rule, 1:1 IoU).
#
# step5_train.py's training self-AP eval calls compute_masks_2d directly and never links
# across z: its GT (labels_to_flows_3d) renumbers ids independently per z-slice, so only a
# per-slice comparison is apples-to-apples there.
#========================================================================================
import numpy as np
import torch
import cv2
import fastremap
from scipy.ndimage import (find_objects, binary_fill_holes, maximum_filter1d,
                            mean as ndimage_mean)
from cellpose.transforms import make_tiles, average_tiles
from config import FLOW_THRESHOLD, MIN_SIZE_PX


torch_GPU = torch.device('cuda')

def fill_holes_and_remove_small_masks(masks, min_size=15):
    """Fill holes in each mask (binary_fill_holes) and remove masks smaller than min_size voxels."""
    slices = find_objects(masks)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk  = masks[slc] == (i + 1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            elif npix > 0:
                msk = binary_fill_holes(msk)
                masks[slc][msk] = (j + 1)
                j += 1
    return masks

def resize_image(img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR):
    """Resize a (Z,Ly,Lx) volume to target (Ly,Lx) or by scale factor rsz -- z is never
    resampled, only the xy plane is resized (same convention used throughout this pipeline)."""
    if Ly is None:
        if not isinstance(rsz, (list, np.ndarray)):
            rsz = [rsz, rsz]
        Ly = int(img0.shape[-2] * rsz[0])
        Lx = int(img0.shape[-1] * rsz[1])
    Lz = img0.shape[0]
    out = np.zeros((Lz, Ly, Lx), img0.dtype)
    for z in range(Lz):
        out[z] = cv2.resize(img0[z], (Lx, Ly), interpolation=interpolation)
    return out

#========================================================================================
# flow tracing (per z-slice; there is no dZ) -> instances
#
# steps2D_interp/follow_flows_2d walk every foreground pixel along the predicted flow to
# the cell centre it converges on. Turning those converged positions into instances then
# happens per slice (compute_masks_2d); linking the slices into 3D cells is a separate
# pass (stitch.py) -- see the module header.
#========================================================================================

def steps2D_interp(p, dP, niter, device=None):
    """Interpolated 2D pixel tracing via torch.nn.functional.grid_sample (GPU only)."""
    shape = dP.shape[1:]
    if device is None:
        device = torch_GPU
    shapeT = np.array(shape)[[1, 0]].astype('double') - 1   # [W-1, H-1]
    pt = torch.from_numpy(p[[1, 0]].T).double().to(device).unsqueeze(0).unsqueeze(0)  # [1,1,n,2] (x,y)
    im = torch.from_numpy(dP[[1, 0]]).double().to(device).unsqueeze(0)                # [1,2,H,W] (dx,dy)
    for k in range(2):
        im[:, k] *= 2. / shapeT[k]
        pt[:, :, :, k] /= shapeT[k]
    pt = pt * 2 - 1
    for t in range(niter):
        dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)  # [1,2,1,n]
        for k in range(2):
            pt[:, :, :, k] = torch.clamp(pt[:, :, :, k] + dPt[:, k], -1., 1.)
    pt = (pt + 1) * 0.5
    for k in range(2):
        pt[:, :, :, k] *= shapeT[k]
    return pt[:, :, :, [1, 0]].cpu().numpy().squeeze().T  # [2, n]  (y, x)

def follow_flows_2d(dP, mask=None, inds=None, niter=200, use_gpu=True, device=None):
    """Trace 2D pixel dynamics along flow field for a single z-slice (GPU only).
    Returns final positions p [2, H, W]."""
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)
    p = np.array(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij'), dtype=np.float32)
    if inds is None:
        inds = np.array(np.nonzero(np.abs(dP[0]) > 1e-3)).astype(np.int32).T
    if inds.ndim < 2 or inds.shape[0] < 5:
        return p, inds, None
    p[:, inds[:, 0], inds[:, 1]] = steps2D_interp(
        p[:, inds[:, 0], inds[:, 1]], dP, niter, device=device
    )
    return p, inds, None

def get_masks_2d(p, iscell=None, rpad=20, flows=None, threshold=0.4, seed_threshold=10,
                 use_gpu=False, device=None):
    """Convert flow-traced positions to 2D instance masks via histogram peak finding.

    seed_threshold: how many flow-traced pixels must converge into the same histogram
    bin before that bin is recognized as its own seed/cell center (default 10, cellpose's
    own value). Lower = looser (more, weaker convergence clusters get recognized as
    separate seeds); higher = tighter (only strongly-converged clusters count, weak ones
    get absorbed into a neighboring seed's territory during the later expansion step
    instead of forming their own instance)."""
    pflows = []
    edges  = []
    shape0 = p.shape[1:]
    dims   = len(p)
    if iscell is not None:
        inds_mesh = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]), indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds_mesh[i][~iscell]
    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5 - rpad, shape0[i] + .5 + rpad, 1))
    h, _  = np.histogramdd(tuple(pflows), bins=edges)
    hmax  = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)
    seeds = np.nonzero(np.logical_and(h - hmax > -1e-6, h > seed_threshold))
    Nmax  = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]
    pix    = list(np.array(seeds).T)
    shape  = h.shape
    expand = np.nonzero(np.ones((3, 3)))
    for it in range(5):
        for k in range(len(pix)):
            if it == 0:
                pix[k] = list(pix[k])
            newpix, iin = [], []
            for i, e in enumerate(expand):
                epix = e[:, np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix >= 0, epix < shape[i]))
                newpix.append(epix)
            iin    = np.all(tuple(iin), axis=0)
            newpix = tuple(newpix)
            igood  = h[newpix] > 2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if it == 4:
                pix[k] = tuple(pix[k])
    M = np.zeros(h.shape, np.uint32)
    for k in range(len(pix)):
        M[pix[k]] = 1 + k
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]
    uniq, counts = fastremap.unique(M0, return_counts=True)
    bigc = uniq[counts > np.prod(shape0) * 0.4]
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        M0 = fastremap.mask(M0, bigc)
    fastremap.renumber(M0, in_place=True)
    return np.reshape(M0, shape0)

def resize_image_2d(img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR):
    """Resize a single (Ly,Lx) 2D image/mask to target (Ly,Lx) or by scale factor rsz."""
    if Ly is None:
        if not isinstance(rsz, (list, np.ndarray)):
            rsz = [rsz, rsz]
        Ly = int(img0.shape[-2] * rsz[0])
        Lx = int(img0.shape[-1] * rsz[1])
    return cv2.resize(img0, (Lx, Ly), interpolation=interpolation)

def _extend_centers_gpu_2d(neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=None):
    """2D diffusion from mask centers (GPU only) -- used by masks_to_flows_gpu_2d,
    which in turn is only used by flow_error_2d's bad-mask QC below (not for the main
    dY/dX segmentation, which comes straight from the network's own predicted flow)."""
    if device is None:
        device = torch_GPU
    nimg = neighbors.shape[0] // 9
    pt   = torch.from_numpy(neighbors).to(device)
    T    = torch.zeros((nimg, Ly, Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device).long()
    isneigh = torch.from_numpy(isneighbor).to(device)
    for i in range(n_iter):
        T[:, meds[:, 0], meds[:, 1]] += 1                      # inject heat at cell centers
        Tneigh  = T[:, pt[:, :, 0], pt[:, :, 1]]
        Tneigh *= isneigh                                        # block heat from leaking into other masks
        T[:, pt[0, :, 0], pt[0, :, 1]] = Tneigh.mean(axis=1)  # diffuse to neighbors
    T     = torch.log(1. + T)
    grads = T[:, pt[[2, 1, 4, 3], :, 0], pt[[2, 1, 4, 3], :, 1]]  # gradients at (top, bottom, right, left)
    dy    = grads[:, 0] - grads[:, 1]
    dx    = grads[:, 2] - grads[:, 3]
    mu_torch = np.stack((dy.cpu().squeeze(), dx.cpu().squeeze()), axis=-2)
    return mu_torch

def masks_to_flows_gpu_2d(masks, device=None):
    """Recompute the 2D flow field a given instance mask WOULD produce (diffusion from each
    instance's center), for comparison against the network's predicted flow in flow_error_2d."""
    if device is None:
        device = torch_GPU
    Ly0, Lx0 = masks.shape
    Ly, Lx   = Ly0 + 2, Lx0 + 2
    masks_padded = np.zeros((Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1] = masks
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y-1, y+1, y, y, y-1, y-1, y+1, y+1), axis=0)
    neighborsX = np.stack((x, x, x, x-1, x+1, x-1, x+1, x-1, x+1), axis=0)
    neighbors  = np.stack((neighborsY, neighborsX), axis=-1)
    slices  = find_objects(masks)
    centers = np.zeros((masks.max(), 2), 'int')
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            yi = np.nonzero(masks[sr, sc] == (i+1))[0].astype(np.int32) + 1
            xi = np.nonzero(masks[sr, sc] == (i+1))[1].astype(np.int32) + 1
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi - xmed)**2 + (yi - ymed)**2)
            centers[i, 0] = yi[imin] + sr.start
            centers[i, 1] = xi[imin] + sc.start
    neighbor_masks = masks_padded[neighbors[:, :, 0], neighbors[:, :, 1]]
    isneighbor     = neighbor_masks == neighbor_masks[0]
    ext    = np.array([[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices])
    n_iter = 2 * (ext.sum(axis=1)).max()
    mu     = _extend_centers_gpu_2d(neighbors, centers, isneighbor, Ly, Lx, n_iter=n_iter, device=device)
    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y-1, x-1] = mu
    return mu0

def flow_error_2d(maski, dP_net, use_gpu=False, device=None):
    """Per-instance mismatch between the network's predicted flow and the flow that maski's
    shape would itself produce -- high error means the mask likely doesn't match a real cell."""
    if dP_net.shape[1:] != maski.shape:
        return np.zeros(0), None
    dP_masks    = masks_to_flows_gpu_2d(maski, device=device)
    flow_errors = np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += ndimage_mean(
            (dP_masks[i] - dP_net[i] / 5.) ** 2,
            maski, index=np.arange(1, maski.max() + 1)
        )
    return flow_errors, dP_masks

def remove_bad_flow_masks_2d(masks, flows, threshold=0.4, use_gpu=False, device=None):
    """Zero out instances whose flow_error_2d exceeds threshold."""
    merrors, _ = flow_error_2d(masks, flows, use_gpu, device)
    badi = 1 + (merrors > threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks

def _finish_mask_2d(mask, resize, min_size):
    if resize is not None:
        recast = mask.max() > 2**16 - 1
        mask = mask.astype(np.float32 if recast else np.uint16)
        mask = resize_image_2d(mask, resize[0], resize[1], interpolation=cv2.INTER_NEAREST)
        if recast:
            mask = mask.astype(np.uint32)
    elif mask.max() < 2**16:
        mask = mask.astype(np.uint16)
    return fill_holes_and_remove_small_masks(mask, min_size=min_size)


def compute_masks_2d(dP, cellprob, p=None, inds=None, niter=200, mask_threshold=0.0,
                     flow_threshold=0.4, resize=None, min_size=15, seed_threshold=10,
                     use_gpu=False, device=None):
    """Convert a single z-slice's net outputs (dY, dX, cellprob) to a 2D instance mask.
    Ids are local to this slice and carry no meaning across z (see the module header).

    dP            : [2, H, W]  (dY, dX) flow predictions for this slice (5x scale from training loss)
    cellprob      : [H, W]     raw logit (mask_threshold=0.0 -> cells where logit > 0)
    resize        : (Ly, Lx)   upsample output mask back to this resolution (None = keep native resolution)
    flow_threshold: discard masks whose flow_error_2d > threshold (set 0/None to skip; cellpose's flow-error QC)
    min_size      : discard masks smaller than this pixel count
    seed_threshold: passed straight through to get_masks_2d -- see its docstring.
    """
    shape = resize if resize is not None else cellprob.shape
    empty = np.zeros(shape, np.uint16)
    cp_mask = cellprob > mask_threshold
    if not np.any(cp_mask):
        return empty
    if p is None:
        p, inds, _ = follow_flows_2d(dP * cp_mask / 5., mask=cp_mask, inds=inds,
                                     niter=niter, use_gpu=use_gpu, device=device)
        if inds is None or inds.ndim < 2 or inds.shape[0] < 5:
            return empty
    mask = get_masks_2d(p, iscell=cp_mask, flows=dP, seed_threshold=seed_threshold, use_gpu=use_gpu)
    if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
        mask = remove_bad_flow_masks_2d(mask, dP, threshold=flow_threshold,
                                        use_gpu=use_gpu, device=device)
    return _finish_mask_2d(mask, resize, min_size)

#========================================================================================
# CPSAM3D (SAM ViT-L backbone + 3D fusion) inference
#
# CPSAM's position embedding is resampled to fit exactly 256x256 (32x32 token) input,
# so unlike CPnet it can't take an arbitrary-size image whole (just padded). The XY
# plane is cut into 256x256 tiles, each tile's "full Z-stack" is passed through the
# net at once (z fusion is preserved intact within a tile), and overlapping tile
# boundaries are stitched with cellpose's own taper-mask blending
# (make_tiles/average_tiles, cellpose.transforms). Tile coordinates (ysub/xsub) are
# computed only once, independent of z, and reused identically for every z-slice.
#========================================================================================

def _z_windows(Z, z_bsize, z_overlap):
    """Returns a list of (z0, z1) half-open windows of depth z_bsize covering [0, Z),
    with z_overlap-deep overlap between consecutive windows. The last window is pulled
    back to end exactly at Z, so it may overlap its predecessor by more than z_overlap
    when Z isn't an exact multiple of the stride (same convention cellpose's own
    make_tiles uses for a final partial xy tile)."""
    stride = max(1, z_bsize - z_overlap)
    starts = list(range(0, Z - z_bsize + 1, stride))
    if not starts or starts[-1] + z_bsize < Z:
        starts.append(max(0, Z - z_bsize))
    starts = sorted(set(starts))
    return [(s, min(s + z_bsize, Z)) for s in starts]


def run_net_3d_tiled(img, net, device, bsize=256, tile_overlap=0.1, z_bsize=None, z_overlap=16):
    """CPSAM3D-specific. img: [C,Z,Ly,Lx] -> y: [nout,Z,Ly,Lx] (restored to original size).

    z_bsize: if given and < Z, the volume is additionally chunked along z into
    z_bsize-thick windows (z_overlap-thick overlap between consecutive windows, blended
    by averaging) before the existing 256x256 xy tiling runs on each window --
    mirrors config.PATCH_SIZE_Z, analogous to how bsize=256 already re-tiles the xy
    plane to match training regardless of the input's own size. None (default, or
    z_bsize >= Z) runs the whole Z depth in one forward pass.
    """
    C, Z, Ly, Lx = img.shape

    # tile coordinates are computed once based on the z=0 slice (applied identically to every z)
    _, ysub, xsub, Lyp, Lxp = make_tiles(img[:, 0], bsize=bsize, tile_overlap=tile_overlap)
    ntiles = len(ysub)

    # CPSAM's pos_embed is sliced to a FIXED bsize x bsize token grid at construction (see
    # cellpose.vit) -- it cannot accept a smaller tile at all. make_tiles only pads up to
    # bsize when the image needs multiple tiles; for an image already <= bsize in a given
    # axis (e.g. a small block after a large rescale) it returns Lyp/Lxp equal to the
    # image's own (smaller) size instead, which would give a sub-bsize tile and fail inside
    # the ViT on the pos_embed addition. Pad explicitly for that case.
    if Lyp < bsize or Lxp < bsize:
        Lyp, Lxp = max(Lyp, bsize), max(Lxp, bsize)
        ysub = [(y0, y0 + bsize) for y0, _ in ysub]
        xsub = [(x0, x0 + bsize) for x0, _ in xsub]

    padded = np.zeros((C, Z, Lyp, Lxp), np.float32)
    padded[:, :, :Ly, :Lx] = img

    nout = net.nout
    net.eval()

    def run_xy_tiled(z0, z1):
        """Existing xy-tiled forward pass, restricted to z-slices [z0, z1)."""
        zlen = z1 - z0
        y_tiles = np.zeros((ntiles, nout, zlen, bsize, bsize), np.float32)
        with torch.no_grad():
            for t in range(ntiles):
                yy0, yy1 = ysub[t]
                xx0, xx1 = xsub[t]
                tile = padded[:, z0:z1, yy0:yy1, xx0:xx1]   # [C,zlen,bsize,bsize]
                out, _ = net(torch.from_numpy(tile[np.newaxis]).to(device))
                y_tiles[t] = out[0].cpu().numpy()

        yf_z = np.zeros((nout, zlen, Lyp, Lxp), np.float32)
        for zi in range(zlen):
            yf_z[:, zi] = average_tiles(y_tiles[:, :, zi], ysub, xsub, Lyp, Lxp)
        return yf_z

    if z_bsize is None or z_bsize >= Z:
        yf = run_xy_tiled(0, Z)
    else:
        yf_sum = np.zeros((nout, Z, Lyp, Lxp), np.float32)
        yf_wt = np.zeros(Z, np.float32)
        for z0, z1 in _z_windows(Z, z_bsize, z_overlap):
            yf_sum[:, z0:z1] += run_xy_tiled(z0, z1)
            yf_wt[z0:z1] += 1
        yf = yf_sum / yf_wt[None, :, None, None]

    return yf[:, :, :Ly, :Lx]


def dx_to_circ(dP):
    """dP = [dy, dx], RAW (not unit-normalized) flow -- same as cellpose.plot.dx_to_circ,
    which cellpose itself calls on the raw predicted dP. Cellpose's own
    sinebow color wheel: brightness comes from this function's own percentile-99
    magnitude normalization below, so background/noise (genuinely near-zero raw flow)
    stays dark and only voxels with real, strong flow light up; hue reflects direction."""
    mag = np.sqrt(np.sum(dP**2, axis=0))
    mag = 255 * np.clip(mag / (np.percentile(mag, 99) + 1e-9), 0, 1.)
    angles = np.arctan2(dP[1], dP[0]) + np.pi
    a = 2
    mag = mag / a
    rgb = np.zeros((*dP.shape[1:], 3))
    rgb[..., 0] = np.clip(mag * (np.cos(angles) + 1), 0, 255)
    rgb[..., 1] = np.clip(mag * (np.cos(angles + 2 * np.pi / 3) + 1), 0, 255)
    rgb[..., 2] = np.clip(mag * (np.cos(angles + 4 * np.pi / 3) + 1), 0, 255)
    return rgb.astype(np.uint8)


def flow_stack_to_rgb(dY, dX, mask=None):
    """dY, dX: (Z,H,W) -> (Z,H,W,3) uint8 sinebow RGB stack. Passes dY,dX straight into
    dx_to_circ RAW, matching cellpose's own call convention -- unit-normalizing them
    first would throw away the relative-magnitude information dx_to_circ's brightness
    scaling depends on, making background noise light up exactly as bright as real
    cell-boundary flow. (masks_to_flows_gpu's unit normalization is not analogous: it only
    applies to known-foreground GT vectors, where the background is exactly 0.)"""
    Z, H, W = dY.shape
    out = np.zeros((Z, H, W, 3), np.uint8)
    for z in range(Z):
        rgb = dx_to_circ(np.stack([dY[z], dX[z]], axis=0))
        if mask is not None:
            rgb[mask[z] == 0] = 40
        out[z] = rgb
    return out


def run_inference_cpsam3d(net, data, device, mask_threshold=0.0, bsize=256, tile_overlap=0.1,
                          rescale_factor=1.0, z_bsize=None, z_overlap=16, return_flow=False,
                          seed_threshold=10, flow_threshold=FLOW_THRESHOLD, min_size=MIN_SIZE_PX):
    """Run trained CPSAM3D on a list of (C,Z,H,W) volumes -> list of PER-SLICE instance masks
    ((Z,H,W) each, compute_masks_2d output stacked, ids local to each slice). This is what
    the review works on (hitl.py's view masks, 3D_segmentation.py's assembled volume); linking the
    slices across z happens separately, once, on the assembled volume (stitch.py).

    flow_threshold / min_size: compute_masks_2d's flow QC and smallest kept piece (px at the
    original input resolution) -- config.FLOW_THRESHOLD / config.MIN_SIZE_PX.

    return_flow=True: returns (masks2d_pred, flows_pred) -- flows_pred a parallel list of
    the (dY,dX) prediction, already upsampled to the original input resolution (see
    rescale_factor note below), for flow_stack_to_rgb visualization.

    z_bsize/z_overlap: passed straight through to run_net_3d_tiled -- chunks any input
    deeper than z_bsize (config.PATCH_SIZE_Z) into z_bsize-thick windows to match what
    the model was actually trained on. None (default) keeps the whole Z depth in one
    forward pass.

    rescale_factor != 1.0: downscales each image by 1/rescale_factor in XY before
    running the net (config.RESCALE_FACTOR, for objects too big for cellpose-SAM's
    256 px tiles). The predicted (dY,dX,cellprob) then gets upsampled back to the original
    resolution BEFORE boundary-tracing/mask computation (bilinear) -- matching
    cellpose's own default (resample=True) behavior -- rather than tracing at the
    low rescaled resolution and nearest-neighbor-upscaling an already-discrete mask
    afterward, which produces blocky/staircase edges (cellpose's own resample=False
    fallback path). Callers always get a mask at the original input resolution
    regardless of rescale_factor. Tracing at the upsampled resolution also needs
    proportionally more flow-following iterations (niter) to actually converge --
    handled below (see the niter computation before the z-slice loop), matching
    cellpose's own niter = 200 / rescale.

    No SizeModel/diameter rescale beyond the above -- CPSAM (cellpose 4.x)
    has deprecated the diam_mean concept itself, and style is a dummy value this net
    doesn't actually compute.
    """
    net.eval()
    use_gpu = (device.type == 'cuda')

    masks2d_pred, flows_pred = [], []
    for i, img in enumerate(data):
        H_orig, W_orig = img.shape[2], img.shape[3]

        if rescale_factor != 1.0:
            rsz = 1.0 / rescale_factor
            img = np.stack([resize_image(img[c], rsz=rsz, interpolation=cv2.INTER_LINEAR)
                            for c in range(img.shape[0])], axis=0)

        y = run_net_3d_tiled(img, net, device, bsize=bsize, tile_overlap=tile_overlap,
                             z_bsize=z_bsize, z_overlap=z_overlap)
        dP       = y[:2]    # [dY, dX]
        cellprob = y[2]

        if rescale_factor != 1.0:
            # upsample the CONTINUOUS flow field back to original resolution BEFORE
            # tracing boundaries (bilinear) -- matches cellpose's own default
            # (resample=True) behavior, which traces at full resolution instead of
            # nearest-neighbor-upscaling an already-discrete low-res mask afterward
            # (blocky/staircase edges, worse the larger rescale_factor is).
            dP = np.stack([resize_image(dP[c], H_orig, W_orig, interpolation=cv2.INTER_LINEAR)
                           for c in range(2)], axis=0)
            cellprob = resize_image(cellprob, H_orig, W_orig, interpolation=cv2.INTER_LINEAR)

        if return_flow:
            flows_pred.append(dP.copy())

        # flow-following (follow_flows_2d) walks each pixel toward its cell center one
        # step at a time; upsampling dP by rescale_factor also multiplies the pixel
        # distance every point needs to walk by that same factor, so niter must scale
        # up with it too or convergence falls short -- matches cellpose's own
        # `niter = int(200/rescale)` (cellpose's rescale = 1/RESCALE_FACTOR here, so
        # niter = int(200*rescale_factor); reduces to the plain default 200 when
        # rescale_factor == 1.0).
        niter = int(200 * rescale_factor)

        masks2d = np.stack([
            compute_masks_2d(
                dP[:, z], cellprob[z],
                niter=niter,
                mask_threshold=mask_threshold,
                resize=None,
                min_size=min_size,
                seed_threshold=seed_threshold,
                flow_threshold=flow_threshold,
                use_gpu=use_gpu,
                device=device,
            )
            for z in range(cellprob.shape[0])
        ])
        print(f'[{i + 1}/{len(data)}] per-slice masks done')
        masks2d_pred.append(masks2d)

    return (masks2d_pred, flows_pred) if return_flow else masks2d_pred

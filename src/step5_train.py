import os
import numpy as np
import cv2
import torch
import scipy.ndimage

from utils.utils import average_precision
from step6_inference_and_postprocessing import compute_masks_2d, run_net_3d_tiled
from config import FLOW_THRESHOLD, MIN_SIZE_PX, RESCALE_FACTOR

# IoU threshold for the per-epoch AP self-eval below (logged as AP@<this>).
AP_IOU_THRESHOLD = 0.5


#========================================================================================
# on-the-fly augmentation (3D -- includes a whole-stack z-flip)
#========================================================================================

def original_random_rotate_and_resize(X, Y=None, scale_range=1., xy=(224, 224),
                                      do_flip=True, rescale=None, unet=False, aux=None):
    """Random rotation + resize augmentation (3D).

    The in-plane (xy) transform -- rotation, horizontal flip, scale, crop -- is built
    once per volume and applied identically to every z-slice (z axis itself is never
    rotated or cropped). On top of that, an independent whole-stack z-flip (reverses
    z-slice order) is applied -- no channel needs a sign correction for it since there
    is no dZ channel.

    X: list of [nchan, Z, Ly, Lx]  normalized volumes
    Y: list of [3, Z, Ly, Lx]      [cellprob, dY, dX]  (mask channel stripped before passing)
    rescale: array [nimg]        diam / diam_mean  (cell size correction factor)
    aux: optional list of [1, Z, Ly, Lx] extra per-item arrays (e.g. the raw instance-id
        mask) that need the exact same crop/rotate/flip/zflip applied (nearest-neighbor,
        like the mask channel) but must NOT go through Y's flow-specific sign-flip/
        rotation math -- kept fully separate from Y so callers that don't pass aux see
        identical behavior to before.

    Returns:
        imgi:    [nimg, nchan, Z, 224, 224]
        lbl:     [nimg, 3, Z, 224, 224]
        scale:   [nimg]
        aux_out: [nimg, 1, Z, 224, 224], or None if aux is None
    """
    scale_range = max(0, min(2, float(scale_range)))
    nimg = len(X)
    nchan = X[0].shape[0] if X[0].ndim > 3 else 1
    Lz = X[0].shape[1]
    imgi = np.zeros((nimg, nchan, Lz, xy[0], xy[1]), np.float32)

    lbl = []
    if Y is not None:
        nt = Y[0].shape[0] if Y[0].ndim > 3 else 1
        lbl = np.zeros((nimg, nt, Lz, xy[0], xy[1]), np.float32)

    aux_out = np.zeros((nimg, 1, Lz, xy[0], xy[1]), np.float32) if aux is not None else None

    scale = np.zeros(nimg, np.float32)

    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]

        # random augmentation parameters
        flip  = np.random.rand() > .5
        zflip = np.random.rand() > .5
        theta = np.random.rand() * np.pi * 2
        scale[n] = (1 - scale_range / 2) + scale_range * np.random.rand()
        if rescale is not None:
            scale[n] *= 1. / rescale[n]   # normalize cell size to diam_mean

        dxy = np.maximum(0, np.array([Lx * scale[n] - xy[1], Ly * scale[n] - xy[0]]))
        dxy = (np.random.rand(2,) - .5) * dxy

        # build affine transform matrix
        cc  = np.array([Lx / 2, Ly / 2])
        cc1 = cc - np.array([Lx - xy[1], Ly - xy[0]]) / 2 + dxy
        pts1 = np.float32([cc, cc + np.array([1, 0]), cc + np.array([0, 1])])
        pts2 = np.float32([
            cc1,
            cc1 + scale[n] * np.array([np.cos(theta),          np.sin(theta)]),
            cc1 + scale[n] * np.array([np.cos(np.pi/2 + theta), np.sin(np.pi/2 + theta)])
        ])
        M = cv2.getAffineTransform(pts1, pts2)

        img = X[n].copy()
        if Y is not None:
            labels = Y[n].copy()
            if labels.ndim < 4:
                labels = labels[np.newaxis, :, :, :]
        if aux is not None:
            auxn = aux[n].copy()
            if auxn.ndim < 4:
                auxn = auxn[np.newaxis, :, :, :]

        # whole-stack z-flip (independent of the in-plane flip/rotation) -- just
        # reverses z-slice order; no channel sign flip needed since there's no dZ
        if zflip and do_flip:
            img = img[:, ::-1]
            if Y is not None:
                labels = labels[:, ::-1]
            if aux is not None:
                auxn = auxn[:, ::-1]

        # horizontal flip
        if flip and do_flip:
            img = img[..., ::-1]
            if Y is not None:
                labels = labels[..., ::-1]
                if nt > 1 and not unet:
                    labels[2] = -labels[2]   # negate dX sign
            if aux is not None:
                auxn = auxn[..., ::-1]

        # warp image -- same 2D affine applied independently to every z-slice
        for k in range(nchan):
            for z in range(Lz):
                imgi[n, k, z] = cv2.warpAffine(img[k, z], M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR)

        # warp labels
        if Y is not None:
            for k in range(nt):
                flag = cv2.INTER_NEAREST if k == 0 else cv2.INTER_LINEAR
                for z in range(Lz):
                    lbl[n, k, z] = cv2.warpAffine(labels[k, z], M, (xy[1], xy[0]), flags=flag)

            # rotate in-plane flow vectors (dY,dX) to match rotation angle
            if nt > 1 and not unet:
                v1 = lbl[n, 2].copy()   # dX
                v2 = lbl[n, 1].copy()   # dY
                lbl[n, 1] = -v1 * np.sin(-theta) + v2 * np.cos(-theta)   # rotated dY
                lbl[n, 2] =  v1 * np.cos(-theta) + v2 * np.sin(-theta)   # rotated dX

        # warp aux (e.g. instance-id mask) -- nearest-neighbor, same M, no flow math
        if aux is not None:
            for z in range(Lz):
                aux_out[n, 0, z] = cv2.warpAffine(auxn[0, z], M, (xy[1], xy[0]), flags=cv2.INTER_NEAREST)

    return imgi, lbl, scale, aux_out


def random_rotate_and_resize(X, Y=None, scale_range=1., xy=(224, 224),
                              do_flip=True, rescale=None, unet=False,
                              min_train_masks=0, max_retries=20, patch_size_z=None, **kwargs):
    """Wrapper for original_random_rotate_and_resize.

    Accepts Y=[train_flows], strips mask channel (index 0), and passes remainder to augmentation.
    train_flows[i] = [mask, cellprob, dY, dX]
    -> y[1:] = [cellprob, dY, dX] only — passed to augmentation

    patch_size_z: if given and smaller than an item's full Z depth, a random
    patch_size_z-thick z-window is drawn from within that item before the xy
    augmentation runs -- the z-axis analogue of how xy always crops down from a larger
    input to `xy`. This is what lets a REVIEW_SIZE_Z-thick HITL patch and a smaller
    PATCH_SIZE_Z-thick model input genuinely differ (config.py). None (default) keeps
    each item's full Z depth, unchanged from before.

    min_train_masks>0: per-crop retry instead of the old whole-patch filter -- a patch
    can easily clear a low whole-patch cell count while most of its 1024x1024 area is
    still empty background, so a crop drawn from a "kept" patch could still land on a
    cell-free region. Each item in the batch is drawn (using the mask channel, warped
    identically via original_random_rotate_and_resize's aux=) and, if the crop's total
    cell count (summed per z-slice, since labels_to_flows_3d renumbers each z-slice
    independently -- the same id on two slices is two different cells) is below
    min_train_masks, redrawn from scratch (new random crop position/rotation/flip, AND a
    new random z-window when patch_size_z is set) up to max_retries times. After
    max_retries the last draw is used regardless, so a patch with genuinely few cells
    anywhere still terminates.

    Always returns (imgi, lbl, scale, aux): aux is the raw per-pixel instance-id mask
    after the same crop/rotate/flip (nearest-neighbor, no flow-specific sign/rotation
    math applied), [nimg, 1, Z, xy, xy], or None when min_train_masks<=0 (that path
    doesn't track it). Used by train_model to build GT-based masks (e.g. for a
    boundary-focused loss term) without re-deriving the crop geometry separately.
    """
    def z_window(x_item, y_item=None):
        Lz = x_item.shape[1]
        if patch_size_z is None or patch_size_z >= Lz:
            return x_item, y_item
        z0 = np.random.randint(0, Lz - patch_size_z + 1)
        x_out = x_item[:, z0:z0 + patch_size_z]
        y_out = y_item[:, z0:z0 + patch_size_z] if y_item is not None else None
        return x_out, y_out

    if Y is None or min_train_masks <= 0:
        if Y is not None:
            cropped = [z_window(X[i], Y[i]) for i in range(len(X))]
            X_z = [x for x, _ in cropped]
            Y_z = [y[1:] for _, y in cropped]
        else:
            X_z = [z_window(x)[0] for x in X]
            Y_z = None
        imgi, lbl, scale, _ = original_random_rotate_and_resize(
            X_z, Y=Y_z, scale_range=scale_range, xy=xy, do_flip=do_flip, rescale=rescale, unet=unet)
        return imgi, lbl, scale, None

    nimg = len(X)
    imgi_list, lbl_list, scale_list = [None] * nimg, [None] * nimg, np.zeros(nimg, np.float32)
    aux_list = [None] * nimg

    for n in range(nimg):
        rescale_n = [rescale[n]] if rescale is not None else None
        for attempt in range(max_retries):
            X_n, Y_n = z_window(X[n], Y[n])
            imgi_n, lbl_n, scale_n, aux_n = original_random_rotate_and_resize(
                [X_n], Y=[Y_n[1:]], scale_range=scale_range, xy=xy,
                do_flip=do_flip, rescale=rescale_n, unet=unet, aux=[Y_n[0:1]])
            n_cells = sum(np.count_nonzero(np.unique(aux_n[0, 0, z]))
                          for z in range(aux_n.shape[2]))
            if n_cells >= min_train_masks or attempt == max_retries - 1:
                break
        imgi_list[n], lbl_list[n], scale_list[n] = imgi_n[0], lbl_n[0], scale_n[0]
        aux_list[n] = aux_n[0]

    imgi = np.stack(imgi_list, axis=0)
    lbl = np.stack(lbl_list, axis=0)
    aux = np.stack(aux_list, axis=0)
    return imgi, lbl, scale_list, aux


def own_boundary_mask(aux, dilate_px=5):
    """aux: (B,1,Z,H,W) raw instance-id labels (int, from random_rotate_and_resize's own aux output).
    Returns (B,Z,H,W) bool mask, True along every non-zero instance's own outline (where its label
    meets anything else, background included), widened by dilate_px but kept inside the foreground:
    the band the z-smoothness term is averaged over. Replaced the contact-only band (boundary
    between two DIFFERENT instances) on 2026-09-27 after a side-by-side comparison in webKnossos."""
    B, _, Z, H, W = aux.shape
    mask = np.zeros((B, Z, H, W), dtype=bool)
    for b in range(B):
        for z in range(Z):
            L = aux[b, 0, z]
            boundary = np.zeros_like(L, dtype=bool)
            boundary[1:, :] |= (L[1:, :] != L[:-1, :])
            boundary[:-1, :] |= (L[1:, :] != L[:-1, :])
            boundary[:, 1:] |= (L[:, 1:] != L[:, :-1])
            boundary[:, :-1] |= (L[:, 1:] != L[:, :-1])
            if dilate_px > 0:
                boundary = scipy.ndimage.binary_dilation(boundary, iterations=dilate_px)
            mask[b, z] = boundary & (L != 0)
    return mask


def cpsam3d_loss(y, lbl, aux, criterion, criterion2, device,
                 zsmooth_boundary_weight_flow=0.0, zsmooth_boundary_weight_cellprob=0.0,
                 zsmooth_boundary_dilate_px=5):
    """The training loss on one batch. y: (B,3,Z,H,W) net output [dY, dX, cellprob logit];
    lbl: (B,3,Z,H,W) numpy targets [cellprob, dY, dX] (random_rotate_and_resize output);
    aux: (B,1,Z,H,W) numpy instance ids or None."""
    veci = 5. * torch.from_numpy(lbl[:, 1:3]).to(device)
    lbl_bin = torch.from_numpy((lbl[:, 0] > 0.5).astype(np.float32)).to(device)
    loss_total = criterion(y[:, :2], veci) / 2. + criterion2(y[:, 2], lbl_bin)

    if (zsmooth_boundary_weight_flow > 0 or zsmooth_boundary_weight_cellprob > 0) and aux is not None:
        boundary = own_boundary_mask(aux, dilate_px=zsmooth_boundary_dilate_px)
        boundary_pair = boundary[:, 1:] | boundary[:, :-1]   # (B, Z-1, H, W)
        boundary_t = torch.from_numpy(boundary_pair).to(device)
        n_mask_px = boundary_t.sum().clamp(min=1)

        if zsmooth_boundary_weight_flow > 0:
            diff_flow = (y[:, :2, 1:] - y[:, :2, :-1]).abs()
            lz_flow_b = (diff_flow * boundary_t.unsqueeze(1)).sum() / (n_mask_px * 2)
            loss_total = loss_total + zsmooth_boundary_weight_flow * lz_flow_b
        if zsmooth_boundary_weight_cellprob > 0:
            diff_cp = (y[:, 2, 1:] - y[:, 2, :-1]).abs()
            lz_cp_b = (diff_cp * boundary_t).sum() / n_mask_px
            loss_total = loss_total + zsmooth_boundary_weight_cellprob * lz_cp_b
    return loss_total


def train_model(net, device, train_flows, train_data,
                 n_epochs=500, learning_rate=1e-5, weight_decay=0.1,
                 batch_size=4, scale_range=0.5, min_train_masks=5,
                 crops_per_epoch=500, freeze_encoder=True, unfreeze_last_n_blocks=0,
                 checkpoint_dir='../models', save_every=100, eval_every=10,
                 patch_size_z=None, zsmooth_boundary_weight_flow=0.0,
                 zsmooth_boundary_weight_cellprob=0.0, zsmooth_boundary_dilate_px=5):
    """Fine-tune CPSAM3D on 3D volumes with on-the-fly augmentation.

    freeze_encoder=True, unfreeze_last_n_blocks=0: freeze the whole encoder, train
    only 3D fusion + readout. freeze_encoder=True, unfreeze_last_n_blocks=N: freeze
    the rest but additionally unfreeze only the encoder's last N transformer blocks.
    Unfreezing all 24 blocks OOMs on this GPU (16GB) even at batch_size=1 (measured),
    so freeze_encoder=False (unfreeze everything) can't be used without gradient
    checkpointing.

    No diam_mean-based size rescale -- matching cellpose-SAM's official recipe
    (`paper/cpsam/train_subsets.py`'s `rescale = not transformer`, and
    `cellpose/train.py`'s `train_seg(rescale=False)` default), the ViT-based CPSAM is
    trained to directly adapt to varying cell sizes without a SizeModel/style vector.

    No test GT mask exists yet, so AP is evaluated on the training volumes themselves. This is
    NOT a generalization metric (same data used for training) -- it's a sanity check that
    the net -> compute_masks_2d pipeline works, and a rough signal for overfitting.

    crops_per_epoch: number of independently-augmented 256x256 crops sampled (with
    replacement across train_data_f) per epoch, grouped into batches of batch_size.

    patch_size_z: passed straight through to random_rotate_and_resize -- draws a random
    patch_size_z-thick z-window from each item's full z depth per crop (config.PATCH_SIZE_Z).
    None (default) keeps each item's full z depth, unchanged from before.

    zsmooth_boundary_weight_flow/zsmooth_boundary_weight_cellprob: L1 penalty between
    the net's own predictions (not GT) at adjacent z-slices, averaged ONLY over a band
    along each GT instance's own outline (own_boundary_mask), not the whole image -- a
    uniform whole-volume mean would dilute the signal into the much larger already-smooth
    background. zsmooth_boundary_dilate_px is the band's width, measured inward from the
    outline. (Until 2026-09-27 the band covered only contacts between two DIFFERENT
    instances; see config.py.)
    """
    net.set_encoder_trainable(not freeze_encoder, unfreeze_last_n_blocks=unfreeze_last_n_blocks)

    # min_train_masks is no longer a whole-patch filter (a 1024x1024 patch can easily
    # clear a low whole-patch cell count while most of its area is still empty
    # background) -- it's enforced per-crop instead, inside random_rotate_and_resize
    # below (redraws a crop if it lands on too few cells).
    train_data_f  = train_data
    train_flows_f = train_flows

    nimg = len(train_data_f)
    print(f'training images: {nimg}')

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in net.parameters())
    print(f'trainable params: {n_trainable:,} / {n_total:,} '
          f'(freeze_encoder={freeze_encoder}, unfreeze_last_n_blocks={unfreeze_last_n_blocks})')

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    criterion  = torch.nn.MSELoss(reduction='mean')       # Flow MSE
    criterion2 = torch.nn.BCEWithLogitsLoss(reduction='mean')  # CellProb BCE

    # learning rate schedule -- 10 epoch warmup then held constant at learning_rate for
    # the rest of training (no end-of-training decay). Cellpose's own train_seg
    # schedule halves the LR repeatedly over the final stretch of training; that decay
    # was cutting each HITL round's effective learning short (round-over-round loss
    # kept stalling even as n_epochs increased), so it's dropped here.
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    LR = LR[:n_epochs]

    n_batches_per_epoch = max(1, int(np.ceil(crops_per_epoch / batch_size)))

    os.makedirs(checkpoint_dir, exist_ok=True) if checkpoint_dir else None

    use_gpu = (device.type == 'cuda')

    loss_log = []
    ap_log   = []   # list of (epoch, AP at AP_IOU_THRESHOLD)

    for iepoch in range(n_epochs):

        for g in optimizer.param_groups:
            g['lr'] = LR[iepoch]

        np.random.seed(iepoch)

        lavg, nsum = 0, 0

        for ibatch in range(n_batches_per_epoch):
            inds = np.random.randint(0, nimg, size=batch_size)

            imgi, lbl, scale, aux = random_rotate_and_resize(
                [train_data_f[i] for i in inds],
                Y=[train_flows_f[i] for i in inds],
                scale_range=scale_range,
                xy=(256, 256),
                unet=False,
                min_train_masks=min_train_masks,
                patch_size_z=patch_size_z,
            )

            X = torch.from_numpy(imgi).to(device)
            optimizer.zero_grad()
            net.train()
            y = net(X)[0]

            loss_total = cpsam3d_loss(y, lbl, aux, criterion, criterion2, device,
                                      zsmooth_boundary_weight_flow, zsmooth_boundary_weight_cellprob,
                                      zsmooth_boundary_dilate_px)

            loss_total.backward()
            optimizer.step()

            lavg += loss_total.item() * len(inds)
            nsum += len(inds)

        lavg /= nsum
        loss_log.append(lavg)

        # -- AP evaluation (train self-eval) --
        # CPSAM3D's position embedding is fixed to exactly 256x256 (32x32 tokens), so
        # unlike CPnet it can't take an arbitrary-size volume whole (just padded to a
        # multiple of 16) -- our training volumes are 768x1024/1024x1024, so they must
        # go through 256x256 tiling (run_net_3d_tiled).
        #
        # Evaluated per z-slice independently (compute_masks_2d), NOT linked across z like
        # the final inference output (stitch) -- the GT mask channel is per-slice-
        # independently-renumbered (labels_to_flows_3d never links ids across z), so only
        # a per-slice comparison is apples-to-apples here. Deliberate divergence from the
        # inference path, for this sanity metric only. Same flow QC and smallest piece as
        # inference; the training volumes are already at the model's resolution, so the
        # smallest piece (raw px) shrinks by RESCALE_FACTOR**2.
        if (iepoch + 1) % eval_every == 0 or iepoch == n_epochs - 1:
            net.eval()
            ap_vals = []
            with torch.no_grad():
                for img_ev, flow_ev in zip(train_data_f, train_flows_f):
                    yt = run_net_3d_tiled(img_ev, net, device, z_bsize=patch_size_z)
                    dP_ev, cellprob_ev = yt[:2], yt[2]
                    for z in range(dP_ev.shape[1]):
                        mask_z = compute_masks_2d(dP_ev[:, z], cellprob_ev[z],
                                                  flow_threshold=FLOW_THRESHOLD,
                                                  min_size=int(MIN_SIZE_PX / RESCALE_FACTOR ** 2),
                                                  use_gpu=use_gpu, device=device)
                        gt_z = flow_ev[0, z]   # mask channel, this z-slice
                        ap, _, _, _ = average_precision([gt_z.astype(np.int32)], [mask_z],
                                                        threshold=[AP_IOU_THRESHOLD])
                        ap_vals.append(float(ap[0, 0]))
            ap_log.append((iepoch, float(np.mean(ap_vals))))

        # log every 10 epochs and the last one (so the final AP is in the log)
        if iepoch % 10 == 0 or iepoch == n_epochs - 1:
            ap_str = f'  AP@{AP_IOU_THRESHOLD}={ap_log[-1][1]:.3f}' if ap_log else ''
            print(f'[Epoch {iepoch:4d}/{n_epochs}]  loss: {lavg:.4f}  lr: {LR[iepoch]:.7f}{ap_str}')

        # save periodic checkpoint
        if checkpoint_dir and (iepoch % save_every == save_every - 1):
            fname = os.path.join(checkpoint_dir, f'epoch_{iepoch+1:04d}.pth')
            net.save_model(fname)
            print(f'  -> checkpoint saved: {fname}')

    print('training complete')
    return train_data_f, train_flows_f, loss_log, ap_log

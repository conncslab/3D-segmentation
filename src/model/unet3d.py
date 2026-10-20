"""A 3D U-Net that turns cellpose-SAM's own 3-channel output (dY,dX,cellprob) plus
the original image into a refined 3-channel prediction (dY,dX,cellprob). No dZ --
z-continuity is left to IoU-based stitching at inference time, not a learned z-flow.

Block style follows cellpose_v1's own U-Net (refs-code/cellpose_v1/cellpose/
resnet_torch.py: resdown/resup) -- residual blocks with a 1x1 projection shortcut
where the channel count changes, pre-activation ordering (norm -> act -> conv), and
additive (not concatenated) skip connections between encoder and decoder. cellpose's
own ablation (Stringer et al., Nature Methods 2021) found this combination
outperforms a standard (non-residual, concatenation-skip) U-Net for this same
flow-prediction task. Two differences from cellpose_v1: GroupNorm instead of
BatchNorm2d (this project trains at batch_size=2, too small for BatchNorm's running
statistics to be reliable), and no style vector (cellpose_v1's resup injects a global
style vector before its second conv; not used here).

Zero-initialized final layer, so on its own this module outputs exactly zero at
construction. CPSAM3D.forward() adds cellpose-SAM's own (dY,dX,cellprob) back in
directly as a shortcut on top of this module's output -- so the combined model
reproduces cellpose-SAM's own prediction exactly at t=0, and only diverges from it as
training actually reduces loss. Same zero-init-residual convention used by
ControlNet's zero convolutions and LoRA's zero-initialized B matrix, applied here to
let a new module extend a pretrained model's output without discarding what the
pretrained model already does well.
"""
import torch
import torch.nn as nn


def z_pool_schedule(z_crop_size, z_xy_ratio=1.0, n_stages=4, z_min=4):
    """Returns a list of n_stages pool kernels, each (2,2,2) or (1,2,2).

    xy always pools by 2x every stage (its own effective voxel size doubles each
    time). z pools too, but only when doing so still keeps z's effective voxel size
    (in xy-voxel-equivalent units) from exceeding xy's effective voxel size at that
    same stage -- and only while z_crop_size stays >= z_min after halving (a floor
    against pooling an already-thin z axis into nothing).

    z_xy_ratio = physical z-step size / physical xy-pixel size (e.g. z_step=0.5um,
    xy_pixel=0.1um -> 5.0). z_xy_ratio=1.0 (isotropic data) pools z in lockstep with
    xy. A dataset with many densely-sampled z-slices (small z_xy_ratio, a
    high-resolution microscope where consecutive slices are nearly redundant) keeps
    pooling z for longer; a dataset with few coarsely-sampled z-slices (large
    z_xy_ratio) stops pooling z almost immediately, since each z-step already carries
    much more independent physical information than a single xy pixel does.
    """
    schedule = []
    z = z_crop_size
    z_eff = z_xy_ratio
    xy_eff = 1.0
    for _ in range(n_stages):
        xy_eff *= 2
        if z // 2 >= z_min and z_eff * 2 <= xy_eff:
            schedule.append((2, 2, 2))
            z = z // 2
            z_eff *= 2
        else:
            schedule.append((1, 2, 2))
    return schedule


def _group_count(channels, max_groups=32):
    """Largest divisor of channels that is <= max_groups, so GroupNorm always
    accepts it regardless of what base_ch and its doublings work out to."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def _conv3d_preact(in_ch, out_ch, kernel_size=3):
    """norm -> act -> conv (pre-activation), matching cellpose_v1's batchconv."""
    return nn.Sequential(
        nn.GroupNorm(_group_count(in_ch), in_ch),
        nn.GELU(),
        nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
    )


def _conv3d_preact_noact(in_ch, out_ch, kernel_size=1):
    """norm -> conv, no activation -- used only for the projection shortcut,
    matching cellpose_v1's batchconv0."""
    return nn.Sequential(
        nn.GroupNorm(_group_count(in_ch), in_ch),
        nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
    )


class ResDown3D(nn.Module):
    """cellpose_v1-style residual encoder block (resdown, 3D): 4 conv layers in 2
    residual sub-blocks. The first sub-block uses a 1x1 projection shortcut (since
    in_ch != out_ch here); the second is a plain residual (out_ch == out_ch, no
    projection needed)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = _conv3d_preact_noact(in_ch, out_ch, kernel_size=1)
        self.conv0 = _conv3d_preact(in_ch, out_ch)
        self.conv1 = _conv3d_preact(out_ch, out_ch)
        self.conv2 = _conv3d_preact(out_ch, out_ch)
        self.conv3 = _conv3d_preact(out_ch, out_ch)

    def forward(self, x):
        x = self.proj(x) + self.conv1(self.conv0(x))
        x = x + self.conv3(self.conv2(x))
        return x


class ResUp3D(nn.Module):
    """cellpose_v1-style residual decoder block (resup, 3D, no style vector). The
    encoder skip y (already at out_ch channels) is added in right after the first
    conv -- an additive skip, not a concatenation, so y's channel count must already
    match out_ch."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = _conv3d_preact_noact(in_ch, out_ch, kernel_size=1)
        self.conv0 = _conv3d_preact(in_ch, out_ch)
        self.conv1 = _conv3d_preact(out_ch, out_ch)
        self.conv2 = _conv3d_preact(out_ch, out_ch)
        self.conv3 = _conv3d_preact(out_ch, out_ch)

    def forward(self, x, y):
        x = self.proj(x) + self.conv1(self.conv0(x) + y)
        x = x + self.conv3(self.conv2(x))
        return x


class UNet3D(nn.Module):
    """in_channels -> out_channels, full spatial resolution throughout (only z/h/w
    get pooled, never the channel count collapsed to a bottleneck the way the old
    per-slice ViT encoder does). Output is exactly zero at construction.

    forward input: (B, in_channels, Z, H, W). Output: (B, out_channels, Z, H, W).
    """

    def __init__(self, in_channels, out_channels, z_crop_size, z_xy_ratio=1.0,
                 base_ch=16, depth=4, z_min=4):
        super().__init__()
        pool_schedule = z_pool_schedule(z_crop_size, z_xy_ratio, n_stages=depth, z_min=z_min)

        self.in_conv = ResDown3D(in_channels, base_ch)
        self.enc_blocks = nn.ModuleList(
            [ResDown3D(base_ch * 2 ** i, base_ch * 2 ** (i + 1)) for i in range(depth)]
        )
        self.pools = nn.ModuleList(
            [nn.MaxPool3d(kernel_size=k, stride=k) for k in pool_schedule]
        )
        self.dec_blocks = nn.ModuleList(
            [ResUp3D(base_ch * 2 ** (i + 1), base_ch * 2 ** i)
             for i in reversed(range(depth))]
        )
        self.out_conv = nn.Conv3d(base_ch, out_channels, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        x = self.in_conv(x)
        skips = [x]
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(pool(x))
            skips.append(x)
        skips.pop()  # last one is the bottleneck itself, not a skip target

        for dec, skip in zip(self.dec_blocks, reversed(skips)):
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='nearest')
            x = dec(x, skip)  # additive skip, applied inside ResUp3D
        return self.out_conv(x)  # (B, out_channels, Z, H, W) -- exactly zero at init

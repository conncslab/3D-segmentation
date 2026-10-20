"""cellpose-SAM (SAM ViT-L backbone, cpsam_v2 weights the cellpose team fine-tuned),
completely unmodified, feeding into a 3D U-Net.

self.base is a stock cellpose.vit.CPSAM (nout=3, the original dY/dX/cellprob output)
-- no shape changes, no partial-weight-loading, so if this backbone itself ever
becomes a fine-tuning target later, it starts from the exact same state a normal
cellpose fine-tune would. It's run independently per z-slice (same cost as the 2D
case) via its own unmodified forward().

The 3D U-Net (UNet3D, model/unet3d.py) takes the padded input image concatenated
with cellpose-SAM's own (dY,dX,cellprob) output and produces a 3-channel delta
(dY,dX,cellprob). cellpose-SAM's own output is then added back in directly as a
shortcut -- the U-Net's own final layer is zero-init, so at construction the
combined model exactly reproduces cellpose-SAM's own prediction and only diverges
from it as training reduces loss (same zero-init-residual convention ControlNet/LoRA
use to extend a pretrained model without discarding it). No dZ: z-continuity is left
to IoU-based stitching at inference time rather than a learned z-flow.

dtype: cellpose's own train.py also casts a bfloat16 net back to float32 before
training starts (so torch.autocast(dtype=net.dtype) becomes effectively a no-op) and
casts it back afterward -- meaning actual fine-tuning always runs in float32 anyway.
Here we skip that round trip and just fix it to float32 from the start.
"""
import torch
import torch.nn as nn

from cellpose.vit import CPSAM
from cellpose.models import cache_model_path

from model.unet3d import UNet3D


class CPSAM3D(nn.Module):
    """Stock CPSAM (per slice) + UNet3D (3-channel: dY,dX,cellprob).

    forward input: (B, C, Z, H, W) -- the same shape the existing 3D augmentation
    (random_rotate_and_resize) produces. Output: (y, style) -- y: (B, nout, Z, H, W) =
    [dY,dX,cellprob].

    style is a dummy (0) for SizeModel compatibility. The original CPSAM also doesn't
    actually compute a style and returns 0 (cellpose 4.x deprecated diam_mean itself,
    no SizeModel), and this branch likewise runs inference at rescale=1.0 with no
    diameter rescale.
    """

    def __init__(self, ps=8, nout=3, bsize=256, rdrop=0.4, z_crop_size=7, z_xy_ratio=1.0):
        super().__init__()
        self.base = CPSAM(ps=ps, nout=3, bsize=bsize, rdrop=rdrop, dtype=torch.float32)
        self.nout = nout
        self.ps = ps
        self.unet = UNet3D(in_channels=3 + 3, out_channels=nout,
                            z_crop_size=z_crop_size, z_xy_ratio=z_xy_ratio, base_ch=16, depth=4)

        # for SizeModel compatibility (same convention as BaseModel, not actually used)
        self.diam_mean = nn.Parameter(torch.tensor([30.]), requires_grad=False)
        self.diam_labels = nn.Parameter(torch.tensor([30.]), requires_grad=False)

    def load_pretrained_cpsam(self, device='cpu'):
        """Loads the cpsam_v2 checkpoint the cellpose team fine-tuned, in full --
        self.base is a stock nout=3 CPSAM so every tensor (including the readout)
        matches the checkpoint's shapes exactly, unlike a modified-nout backbone.
        strict=False only because the checkpoint itself doesn't include
        diam_mean/diam_labels (BaseModel's own SizeModel-compat dummies, unrelated to
        this project's nout change) -- not a shape mismatch."""
        path = cache_model_path('cpsam_v2')
        sd = torch.load(path, map_location=device, weights_only=True)
        missing, unexpected = self.base.load_state_dict(sd, strict=False)
        assert not unexpected, f'unexpected keys in checkpoint: {unexpected}'
        assert set(missing) <= {'diam_mean', 'diam_labels'}, f'unexpected missing keys: {missing}'
        print(f'[CPSAM3D] pretrained cellpose-SAM 전부 그대로 로드 ({len(sd)}개 텐서, '
              f'diam_mean/diam_labels(SizeModel용, 미사용)만 체크포인트에 없어서 기본값 유지).')

    def forward(self, x):
        # x: (B, C=3, Z, H, W)
        B, C, Z, H, W = x.shape
        x_orig = x
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * Z, C, H, W)

        cpsam_out, _ = self.base(x_flat)                                     # (B*Z, 3, H, W) -- stock CPSAM forward
        cpsam_out = cpsam_out.reshape(B, Z, 3, H, W).permute(0, 2, 1, 3, 4)   # (B, 3, Z, H, W)

        unet_input = torch.cat([x_orig, cpsam_out], dim=1)   # (B, 6, Z, H, W)
        delta = self.unet(unet_input)                        # (B, nout, Z, H, W) -- zero at init

        out = delta + cpsam_out

        style = torch.zeros((B, 256), device=x.device, dtype=x.dtype)
        return out, style

    def set_encoder_trainable(self, trainable, unfreeze_last_n_blocks=0):
        """Stage-1 strategy: freeze the encoder, train only the unet (trainable=False).
        Stage-2: instead of unfreezing everything, additionally train only the last
        unfreeze_last_n_blocks transformer blocks -- unfreezing all 24 blocks OOMs on
        this GPU (16GB) even at batch_size=1 (measured), while unfreezing only the last
        4 blocks fits in 11.7GB at batch_size=2.
        (patch_embed is accessed via .weight.data inside cellpose's own CPSAM.forward,
        so no gradient ever flows through it regardless of requires_grad)"""
        for p in self.base.encoder.parameters():
            p.requires_grad = trainable
        if not trainable and unfreeze_last_n_blocks > 0:
            for blk in self.base.encoder.blocks[-unfreeze_last_n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device=None):
        sd = torch.load(filename, map_location=device or 'cpu', weights_only=True)
        self.load_state_dict(sd)

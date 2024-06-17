import torch.nn as nn
import torch
import numpy as np


class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]}
    ):
        super().__init__()

        dim_total = 0
        dim_max = 0

        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total

        self.encoder = encoder
        self.decoder = decoder
        self.outputs = outputs

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max, 1))

    def forward(self, image, export=False, fixed_IE=False, intrinsics=None, extrinsics=None):
        if torch.is_tensor(export):
            export = export.item()

        if torch.is_tensor(fixed_IE):
            fixed_IE = fixed_IE.item()

        if fixed_IE:
            intrinsics = torch.from_numpy(np.load("/home/ava/rajrup/refer/GKT/segmentation/src/artifacts/intrinsics.npy"))
            extrinsics = torch.from_numpy(np.load("/home/ava/rajrup/refer/GKT/segmentation/src/artifacts/extrinsics.npy"))
            x = self.encoder(image, intrinsics, extrinsics, export, fixed_IE)
        else:
            x = self.encoder(image, intrinsics, extrinsics, export, fixed_IE)

        y = self.decoder(x)
        z = self.to_logits(y)
        
        return z

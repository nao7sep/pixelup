"""The small Real-ESRGAN network subset used by PixelUp.

``RRDBNet`` is adapted from BasicSR (Apache-2.0). ``SRVGGNetCompact`` is
adapted from Real-ESRGAN (BSD-3-Clause). The corresponding notices and license
texts are in the repository's ``THIRD_PARTY_NOTICES`` file.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


@torch.no_grad()
def _default_init_weights(modules: list[nn.Module], scale: float = 1) -> None:
    for module in modules:
        for layer in module.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight)
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.fill_(0)


def _make_layer(
    block: Callable[..., nn.Module],
    count: int,
    **kwargs: int,
) -> nn.Sequential:
    return nn.Sequential(*(block(**kwargs) for _ in range(count)))


def _pixel_unshuffle(tensor: torch.Tensor, scale: int) -> torch.Tensor:
    batch, channels, full_height, full_width = tensor.size()
    output_channels = channels * scale**2
    if full_height % scale != 0 or full_width % scale != 0:
        raise ValueError("Image dimensions are not divisible by the model scale.")
    height = full_height // scale
    width = full_width // scale
    view = tensor.view(batch, channels, height, scale, width, scale)
    return view.permute(0, 1, 3, 5, 2, 4).reshape(
        batch,
        output_channels,
        height,
        width,
    )


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        _default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
            0.1,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output1 = self.lrelu(self.conv1(tensor))
        output2 = self.lrelu(self.conv2(torch.cat((tensor, output1), 1)))
        output3 = self.lrelu(self.conv3(torch.cat((tensor, output1, output2), 1)))
        output4 = self.lrelu(self.conv4(torch.cat((tensor, output1, output2, output3), 1)))
        output5 = self.conv5(torch.cat((tensor, output1, output2, output3, output4), 1))
        return output5 * 0.2 + tensor


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = self.rdb1(tensor)
        output = self.rdb2(output)
        output = self.rdb3(output)
        return output * 0.2 + tensor


class RRDBNet(nn.Module):
    """ESRGAN's residual-in-residual dense-block network."""

    def __init__(
        self,
        num_in_ch: int,
        num_out_ch: int,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = _make_layer(
            RRDB,
            num_block,
            num_feat=num_feat,
            num_grow_ch=num_grow_ch,
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feature = _pixel_unshuffle(tensor, scale=2)
        elif self.scale == 1:
            feature = _pixel_unshuffle(tensor, scale=4)
        else:
            feature = tensor
        feature = self.conv_first(feature)
        body_feature = self.conv_body(self.body(feature))
        feature += body_feature
        feature = self.lrelu(
            self.conv_up1(F.interpolate(feature, scale_factor=2, mode="nearest"))
        )
        feature = self.lrelu(
            self.conv_up2(F.interpolate(feature, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feature)))


class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN's compact VGG-style super-resolution network."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(_activation(act_type, num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(_activation(act_type, num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = tensor
        for layer in self.body:
            output = layer(output)
        output = self.upsampler(output)
        base = F.interpolate(tensor, scale_factor=self.upscale, mode="nearest")
        output += base
        return output


def _activation(act_type: str, num_feat: int) -> nn.Module:
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    if act_type == "prelu":
        return nn.PReLU(num_parameters=num_feat)
    if act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"Unsupported activation: {act_type}")

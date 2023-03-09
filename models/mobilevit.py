import torch.nn as nn
from einops import rearrange
import torch
from typing import Callable, Any, Optional, List
import torch.nn.functional as F


model_cfg = {
    "xxs": {
        "features": [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
        "d": [64, 80, 96],
        "expansion_ratio": 2,
        "layers": [2, 4, 3]
    },
    "xs": {
        "features": [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        "d": [96, 120, 144],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
    "s": {
        "features": [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        "d": [144, 192, 240],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
}


class ConvNormAct(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride=1,
                 padding: Optional[int] = None,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.SiLU,
                 dilation: int = 1
                 ):
        super(ConvNormAct, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                    dilation=dilation, groups=groups, bias=norm_layer is None)

        self.norm_layer = nn.BatchNorm2d(out_channels) if norm_layer is None else norm_layer(out_channels)
        self.act = activation_layer() if activation_layer is not None else activation_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, dim, num_heads=8, dim_head=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        _weight_dim = self.num_heads * self.dim_head
        self.to_qvk = nn.Linear(dim, _weight_dim * 3, bias=False)
        self.scale_factor = dim ** -0.5

        # Weight matrix for output, Size: num_heads*dim_head X dim
        # Final linear transformation layer
        self.w_out = nn.Linear(_weight_dim, dim, bias=False)

    def forward(self, x):
        qkv = self.to_qvk(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale_factor
        attn = torch.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.w_out(out)


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadSelfAttention(dim, heads, dim_head)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MobileVitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_model, layers, mlp_dim):
        super(MobileVitBlock, self).__init__()
        # Local representation
        self.local_representation = nn.Sequential(
            # Encode local spatial information
            ConvNormAct(in_channels, in_channels, 3),
            # Projects the tensor to a high-diementional space
            ConvNormAct(in_channels, d_model, 1)
        )

        self.transformer = Transformer(d_model, layers, 1, 32, mlp_dim, 0.1)

        # Fusion block
        self.fusion_block1 = nn.Conv2d(d_model, in_channels, kernel_size=1)
        self.fusion_block2 = nn.Conv2d(in_channels * 2, out_channels, 3, padding=1)

    def forward(self, x):
        local_repr = self.local_representation(x)
        # global_repr = self.global_representation(local_repr)
        _, _, h, w = local_repr.shape
        # b c h w -> b patch_h*patch_w(windows_size) num_patches d
        global_repr = rearrange(local_repr, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=2, pw=2)
        global_repr = self.transformer(global_repr)
        global_repr = rearrange(global_repr, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // 2, w=w // 2, ph=2, pw=2)

        # Fuse the local and gloval features in the concatenation tensor
        fuse_repr = self.fusion_block1(global_repr)
        result = self.fusion_block2(torch.cat([x, fuse_repr], dim=1))
        return result


class InvertedResidual(nn.Module):
    """
    MobileNetv2 InvertedResidual block
    """

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2, act_layer=nn.SiLU):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvNormAct(in_channels, hidden_dim, kernel_size=1, activation_layer=None))

        # Depth-wise convolution
        layers.append(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                        groups=hidden_dim, activation_layer=act_layer)
        )
        # Point-wise convolution
        layers.append(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViT(nn.Module):
    def __init__(self, cfg):
        super(MobileViT, self).__init__()

        model_config = model_cfg[cfg.model_scale]
        img_size = 256
        num_classes = 1000
        features_list = model_config["features"]
        d_list = model_config["d"]
        transformer_depth = model_config["layers"]
        expansion = model_config["expansion_ratio"]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=features_list[0], kernel_size=3, stride=2, padding=1),
            InvertedResidual(in_channels=features_list[0], out_channels=features_list[1], stride=1,
                             expand_ratio=expansion),
        )

        self.stage1 = nn.Sequential(
            InvertedResidual(in_channels=features_list[1], out_channels=features_list[2], stride=2,
                             expand_ratio=expansion),
            InvertedResidual(in_channels=features_list[2], out_channels=features_list[2], stride=1,
                             expand_ratio=expansion),
            InvertedResidual(in_channels=features_list[2], out_channels=features_list[3], stride=1,
                             expand_ratio=expansion)
        )

        self.stage2 = nn.Sequential(
            InvertedResidual(in_channels=features_list[3], out_channels=features_list[4], stride=2,
                             expand_ratio=expansion),
            MobileVitBlock(in_channels=features_list[4], out_channels=features_list[5], d_model=d_list[0],
                           layers=transformer_depth[0], mlp_dim=d_list[0] * 2)
        )

        self.stage3 = nn.Sequential(
            InvertedResidual(in_channels=features_list[5], out_channels=features_list[6], stride=2,
                             expand_ratio=expansion),
            MobileVitBlock(in_channels=features_list[6], out_channels=features_list[7], d_model=d_list[1],
                           layers=transformer_depth[1], mlp_dim=d_list[1] * 4)
        )

        self.stage4 = nn.Sequential(
            InvertedResidual(in_channels=features_list[7], out_channels=features_list[8], stride=2,
                             expand_ratio=expansion),
            MobileVitBlock(in_channels=features_list[8], out_channels=features_list[9], d_model=d_list[2],
                           layers=transformer_depth[2], mlp_dim=d_list[2] * 4),
            nn.Conv2d(in_channels=features_list[9], out_channels=features_list[10], kernel_size=1, stride=1, padding=0)
        )

        self.avgpool = nn.AvgPool2d(kernel_size=img_size // 32)
        self.fc = nn.Linear(features_list[10], num_classes)

    def forward(self, x):
        # Stem
        x = self.stem(x)
        # Body
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # Head
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MobileViTDecoder(nn.Module):
    def __init__(self, cfg):
        super(MobileViTDecoder, self).__init__()
        if cfg.model_scale == "xxs":
            self.channel = [24, 48, 64, 320]
            d_list = model_cfg["xxs"]["d"]
            layers = model_cfg["xxs"]["layers"]
        elif cfg.model_scale == "xs":
            self.channel = [48, 64, 80, 384]
            d_list = model_cfg["xs"]["d"]
            layers = model_cfg["xs"]["layers"]
        elif cfg.model_scale == "s":
           self.channel = [64, 96, 128, 640]
           d_list = model_cfg["s"]["d"]
           layers = model_cfg["s"]["layers"]
        else:
            raise NotImplementedError

        self.stage1 = nn.Sequential(
            MobileVitBlock(in_channels=self.channel[3], out_channels=self.channel[3], d_model=d_list[2], layers=layers[2],
                           mlp_dim=d_list[2] * 4),
            ConvNormAct(in_channels=self.channel[3], out_channels=self.channel[2], kernel_size=1)
        )
        self.stage2 = nn.Sequential(
            MobileVitBlock(in_channels=self.channel[2], out_channels=self.channel[2], d_model=d_list[1], layers=layers[1],
                           mlp_dim=d_list[1] * 4),
            ConvNormAct(in_channels=self.channel[2], out_channels=self.channel[1], kernel_size=1)
        )
        self.stage3 = nn.Sequential(
            MobileVitBlock(in_channels=self.channel[1], out_channels=self.channel[1], d_model=d_list[0], layers=layers[0],
                           mlp_dim=d_list[0] * 4),
            ConvNormAct(in_channels=self.channel[1], out_channels=self.channel[0], kernel_size=1)
        )

        self.dropout = nn.Dropout(p=cfg.dropout) if cfg.dropout !=0 else nn.Identity()
        self.conv_seg = nn.Conv3d(self.channel[0], 1, kernel_size=3, padding=1, stride=1)

        self.r1 = ConvNormAct(in_channels=self.channel[2], out_channels=self.channel[2], kernel_size=3, padding=1)
        self.r2 = ConvNormAct(in_channels=self.channel[1], out_channels=self.channel[1], kernel_size=3, padding=1)
        self.r3 = ConvNormAct(in_channels=self.channel[0], out_channels=self.channel[0], kernel_size=3, padding=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(0)
                m.bias.data.zero_()
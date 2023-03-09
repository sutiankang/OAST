import torch
import torch.nn as nn
from models.inflate import inflate_conv, inflate_batch_norm
from models.mobilevit import MobileViT, MobileViTDecoder
from collections import OrderedDict
import torch.nn.functional as F
from einops import rearrange


class ConvNormAct3D(nn.Module):
    def __init__(self, ConvNormAct):
        super(ConvNormAct3D, self).__init__()

        self.conv = inflate_conv(ConvNormAct.conv, time_dim=3, time_padding=1, time_stride=1, center=True)
        self.norm_layer = inflate_batch_norm(ConvNormAct.norm_layer)
        self.act = ConvNormAct.act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.act is not None:
            x = self.act(x)
        return x


class InvertedResidual3D(nn.Module):
    def __init__(self, InvertedResidual):
        super(InvertedResidual3D, self).__init__()
        self.stride = InvertedResidual.stride
        self.use_res_connect = InvertedResidual.use_res_connect
        self.conv = nn.Sequential(
            ConvNormAct3D(InvertedResidual.conv[0]),
            ConvNormAct3D(InvertedResidual.conv[1]),
            inflate_conv(InvertedResidual.conv[2], time_dim=3, time_padding=1, time_stride=1, center=True),
            inflate_batch_norm(InvertedResidual.conv[3])
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Transformer3D(nn.Module):
    def __init__(self, Transformer):
        super(Transformer3D, self).__init__()
        self.layers = Transformer.layers

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MobileVitBlock3D(nn.Module):
    def __init__(self, MobileVitBlock):
        super(MobileVitBlock3D, self).__init__()
        self.local_representation = nn.Sequential(
            ConvNormAct3D(MobileVitBlock.local_representation[0]),
            ConvNormAct3D(MobileVitBlock.local_representation[1])
        )

        self.transformer = Transformer3D(MobileVitBlock.transformer)

        # Fusion block
        self.fusion_block1 = inflate_conv(MobileVitBlock.fusion_block1, time_dim=3, time_padding=1, time_stride=1,
                                          center=True)
        self.fusion_block2 = inflate_conv(MobileVitBlock.fusion_block2, time_dim=3, time_stride=1, time_padding=1,
                                          center=True)

    def forward(self, x):
        local_repr = self.local_representation(x)
        # global_repr = self.global_representation(local_repr)
        _, _, t, h, w = local_repr.shape
        # b c h w -> b patch_h*patch_w(windows_size) num_patches d
        global_repr = rearrange(local_repr, 'b d (t pt) (h ph) (w pw) -> b (ph pw pt) (h w t) d', ph=2, pw=2, pt=1)
        global_repr = self.transformer(global_repr)
        global_repr = rearrange(global_repr, 'b (ph pw pt) (h w t) d -> b d (t pt) (h ph) (w pw)',
                                h=h // 2, w=w // 2, t=t, ph=2, pw=2, pt=1)
        # Fuse the local and gloval features in the concatenation tensor
        fuse_repr = self.fusion_block1(global_repr)
        result = self.fusion_block2(torch.cat([x, fuse_repr], dim=1))
        return result


class MobileViT3D(nn.Module):
    def __init__(self, MobileViT):
        super(MobileViT3D, self).__init__()
        self.stem = nn.Sequential(
            inflate_conv(MobileViT.stem[0], time_dim=3, time_padding=1, time_stride=1, center=True),
            InvertedResidual3D(MobileViT.stem[1])
        )

        self.stage1 = nn.Sequential(
            InvertedResidual3D(MobileViT.stage1[0]),
            InvertedResidual3D(MobileViT.stage1[1]),
            InvertedResidual3D(MobileViT.stage1[2])
        )

        self.stage2 = nn.Sequential(
            InvertedResidual3D(MobileViT.stage2[0]),
            MobileVitBlock3D(MobileViT.stage2[1])
        )

        self.stage3 = nn.Sequential(
            InvertedResidual3D(MobileViT.stage3[0]),
            MobileVitBlock3D(MobileViT.stage3[1])
        )

        self.stage4 = nn.Sequential(
            InvertedResidual3D(MobileViT.stage4[0]),
            MobileVitBlock3D(MobileViT.stage4[1]),
            inflate_conv(MobileViT.stage4[2], time_dim=3, time_stride=1, time_padding=1, center=True)
        )


class MobileViTVOSEncoder(nn.Module):
    def __init__(self, cfg):
        super(MobileViTVOSEncoder, self).__init__()

        self.cfg = cfg
        model = self._forward_fn()
        self.stem = model[0]
        self.body = nn.Sequential(
            model[1],
            model[2],
            model[3],
            model[4]
        )

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for i in range(len(self.body)):
            x = self.body[i](x)
            outputs.append(x)

        return outputs

    def _forward_fn(self):
        model = MobileViT(self.cfg)
        if self.cfg.pretrained:
            state_dict = torch.load(self.cfg.pretrained, map_location="cpu")["state_dict"]
            for key in list(state_dict.keys()):
                state_dict[key.replace('module.', '')] = state_dict.pop(key)
            model.load_state_dict(state_dict, strict=False)
            print(f"loaded mobilevit_{self.cfg.model_scale} backbone from {self.cfg.pretrained}.")

        model = nn.Sequential(
            MobileViT3D(model).stem,
            MobileViT3D(model).stage1,
            MobileViT3D(model).stage2,
            MobileViT3D(model).stage3,
            MobileViT3D(model).stage4
        )
        return model


class MobileViTVOSDecoder(nn.Module):
    def __init__(self, MobileViTDecoder):
        super(MobileViTVOSDecoder, self).__init__()

        self.stage1 = nn.Sequential(
            MobileVitBlock3D(MobileViTDecoder.stage1[0]),
            ConvNormAct3D(MobileViTDecoder.stage1[1])
        )

        self.stage2 = nn.Sequential(
            MobileVitBlock3D(MobileViTDecoder.stage2[0]),
            ConvNormAct3D(MobileViTDecoder.stage2[1])
        )

        self.stage3 = nn.Sequential(
            MobileVitBlock3D(MobileViTDecoder.stage3[0]),
            ConvNormAct3D(MobileViTDecoder.stage3[1])
        )

        self.r1 = ConvNormAct3D(MobileViTDecoder.r1)
        self.r2 = ConvNormAct3D(MobileViTDecoder.r2)
        self.r3 = ConvNormAct3D(MobileViTDecoder.r3)

        self.dropout = MobileViTDecoder.dropout
        self.conv_seg = MobileViTDecoder.conv_seg
        self.init_weights()

    def forward(self, outputs):

        x = self.stage1(outputs[3])
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True) + self.r1(outputs[2])
        x = self.stage2(x)
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True) + self.r2(outputs[1])
        x = self.stage3(x)
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True) + self.r3(outputs[0])
        x = self.dropout(x)
        x = self.conv_seg(x)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(0)
                m.bias.data.zero_()


class MobileViTVOS(nn.Module):
    def __init__(self, cfg):
        super(MobileViTVOS, self).__init__()
        self.encoder = MobileViTVOSEncoder(cfg)
        self.decoder = MobileViTVOSDecoder(MobileViTDecoder(cfg))

    def forward(self, img, flow):
        x = torch.stack([img, flow], dim=2)
        size = x.shape[-3:]
        x = self.encoder(x)
        # B C T H W
        feature_map = F.interpolate(x[3], size=size, mode="trilinear", align_corners=True).mean(2).mean(1)
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=True)
        # B C T H W - > B C H W
        x = x[:, :, 0]
        return x, feature_map


if __name__ == '__main__':

    x = torch.randn(2, 3, 384, 640)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", default=0)
    parser.add_argument("--model_scale", default="s")
    parser.add_argument("--pretrained", default="")
    cfg = parser.parse_args()

    model = MobileViTVOS(cfg)
    results = model(x, x)
    print(results[0].shape,results[1].shape)


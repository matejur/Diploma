import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.swin_transformer import swin_v2_t, swin_v2_s, swin_v2_b, Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights

from .swin import SwinTransformer, SwinTransformerCrossBlock

class SwinBackbone(nn.Module):
    def __init__(self, backbone="swin_tiny", pretrained=True):
        super().__init__()

        if backbone == "swin_tiny":
            net = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT if pretrained else None)
        elif backbone == "swin_small":
            net = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT if pretrained else None)
        elif backbone == "swin_base":
            net = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT if pretrained else None)
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))

        net = net.features
        self.embed_patches = net[0]

        self.attention_1 = net[1]
        self.merge_1 = net[2]

        self.attention_2 = net[3]
        self.merge_2 = net[4]

        self.attention_3 = net[5]
        self.merge_3 = net[6]

        self.attention_4 = net[7]

    def forward(self, x):

        h, w = x.shape[2], x.shape[3]
        # check input size
        if h % 64 != 0 or w % 64 != 0:
            assert False, "invalid input size: {}".format(x.shape)

        x = self.embed_patches(x)
        out_1 = self.attention_1(x)

        x = self.merge_1(out_1)
        out_2 = self.attention_2(x)

        x = self.merge_2(out_2)
        out_3 = self.attention_3(x)

        x = self.merge_3(out_3)
        out_4 = self.attention_4(x)

        return [out_1, out_2, out_3, out_4]


class SwinBasedCrossFusion(nn.Module):
    def __init__(self, out_dims):
        super().__init__()

        window_size = [8, 8]
        self.attn1 = SwinTransformerCrossBlock(
                dim=out_dims,
                num_heads=8,
                window_size=window_size,
                shift_size=[0, 0],
                mlp_ratio=4,
                dropout=0.5,
                attention_dropout=0.5,
                norm_layer = nn.LayerNorm
            )

        self.attn2 = SwinTransformerCrossBlock(
                dim=out_dims,
                num_heads=8,
                window_size=window_size,
                shift_size=[4, 4],
                mlp_ratio=4,
                dropout=0.5,
                attention_dropout=0.5,
                norm_layer = nn.LayerNorm
            )

    def forward(self, pcd, img):
        # attn(q, kv)
        x = self.attn1(pcd, img)
        x = self.attn2(x, img)
        # mogoče removaj ta skip
        x = x + pcd
        return x


class ConvUpsample(nn.Module):
    def __init__(self, in_channels=[], n_classes=4, base_channels=64):
        super(ConvUpsample, self).__init__()

        self.up_4a = nn.Sequential(
            nn.Conv2d(in_channels[3], base_channels, 3, padding=1),
            nn.LeakyReLU(),
            #nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.norm_4a = nn.LayerNorm(base_channels)

        self.up_3a = nn.Sequential(
            nn.Conv2d(in_channels[2] + base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            #nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.norm_3a = nn.LayerNorm(base_channels)

        self.up_2a = nn.Sequential(
            nn.Conv2d(in_channels[1] + base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            #nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.norm_2a = nn.LayerNorm(base_channels)

        self.up_1a = nn.Sequential(
            nn.Conv2d(in_channels[0] + base_channels, base_channels, 1),
            nn.LeakyReLU(),
            #nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.norm_1a = nn.LayerNorm(base_channels)

        self.up_0a = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1),
            nn.LeakyReLU(),
            #nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.norm_0a = nn.LayerNorm(base_channels)

        self.conv = nn.Conv2d(base_channels, n_classes, kernel_size=3, padding=1)

    def norm(self, x, norm):
        x = x.permute(0, 2, 3, 1)
        x = norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, inputs):
        for i in range(len(inputs)):
            inputs[i] = inputs[i].permute(0, 3, 1, 2)

        up_4a = self.up_4a(inputs[3])
        #up_4a = self.norm(up_4a, self.norm_4a)
        up_3a = self.up_3a(torch.cat((up_4a, inputs[2]), dim=1))
        #up_3a = self.norm(up_3a, self.norm_3a)
        up_2a = self.up_2a(torch.cat((up_3a, inputs[1]), dim=1))
        #up_2a = self.norm(up_2a, self.norm_2a)
        up_1a = self.up_1a(torch.cat((up_2a, inputs[0]), dim=1))
        #up_1a = self.norm(up_1a, self.norm_1a)
        up_0a = self.up_0a(up_1a)
        #up_0a = self.norm(up_0a, self.norm_0a)
        out = self.conv(up_0a)
        out = F.softmax(out, dim=1)
        return out


class SwinCrossFusion(SwinTransformer):
    def __init__(self):
        super().__init__(in_channels=5, patch_size=[4, 4], embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=[8, 8], dropout=0.5, attention_dropout=0.5)
        self.fusion_1 = SwinBasedCrossFusion(out_dims=96)
        self.fusion_2 = SwinBasedCrossFusion(out_dims=192)
        self.fusion_3 = SwinBasedCrossFusion(out_dims=384)
        self.fusion_4 = SwinBasedCrossFusion(out_dims=768)

        net = self.features
        self.embed_patches = net[0]

        self.attention_1 = net[1]
        self.merge_1 = net[2]

        self.attention_2 = net[3]
        self.merge_2 = net[4]

        self.attention_3 = net[5]
        self.merge_3 = net[6]

        self.attention_4 = net[7]

        self.upsample = ConvUpsample(in_channels=[96, 192, 384, 768], n_classes=20, base_channels=96)

    def forward(self, pcd_feature, img_features):
        x = self.embed_patches(pcd_feature)

        out_1 = self.attention_1(x)
        out_1 = self.fusion_1(out_1, img_features[0])
        
        out_2 = self.merge_1(out_1)
        out_2 = self.attention_2(out_2)
        out_2 = self.fusion_2(out_2, img_features[1])

        out_3 = self.merge_2(out_2)
        out_3 = self.attention_3(out_3)
        out_3 = self.fusion_3(out_3, img_features[2])

        out_4 = self.merge_3(out_3)
        out_4 = self.attention_4(out_4)
        out_4 = self.fusion_4(out_4, img_features[3])

        return self.upsample([out_1, out_2, out_3, out_4])

class FusionCrossNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = SwinBackbone(backbone)

        self.fusion = SwinCrossFusion()

        self.upsample = ConvUpsample(in_channels=[96, 192, 384, 768], n_classes=20, base_channels=96)

    def forward(self, pcd_feature, img_feature):
        img_features = self.backbone(img_feature)

        pcd_out = self.fusion(pcd_feature, img_features)

        img_out = self.upsample(img_features)

        return pcd_out, img_out
        
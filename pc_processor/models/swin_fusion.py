import torch
import torch.nn as nn
from torchvision.models.swin_transformer import swin_v2_t, swin_v2_s, swin_v2_b, Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights

from .swin import SwinTransformer, SwinTransformerBlock

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


class SwinBasedFusion(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()

        window_size = [8, 8]
        layers = []
        for i in range(2):
            layers.append(SwinTransformerBlock(
                dim=out_dims,
                num_heads=8,
                window_size=window_size,
                shift_size=[0 if i % 2 == 0 else w // 2 for w in window_size],
                mlp_ratio=4,
                dropout=0.0,
                attention_dropout=0.0,
                stochastic_depth_prob=0.0,
                norm_layer = nn.LayerNorm
            ))

        self.reduce_dim = nn.Sequential(
            nn.Linear(in_dims, 4 * in_dims),
            nn.LeakyReLU(),
            nn.LayerNorm(4 * in_dims),
            nn.Linear(4 * in_dims, out_dims)
        )

        self.attention = nn.Sequential(*layers)
        

    def forward(self, pcd, img):
        skip = pcd
        cat_future = torch.cat([pcd, img], dim=3)
        x = self.reduce_dim(cat_future)
        x = self.attention(x)
        x = x + skip
        return x

class ConvUpsample(nn.Module):
    def __init__(self, in_channels=[], n_classes=20, base_channels=96):
        super(ConvUpsample, self).__init__()

        self.up_4a = nn.Sequential(
            nn.Conv2d(in_channels[3], base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.norm_4a = nn.LayerNorm(base_channels)

        self.up_3a = nn.Sequential(
            nn.Conv2d(in_channels[2] + base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.norm_3a = nn.LayerNorm(base_channels)

        self.up_2a = nn.Sequential(
            nn.Conv2d(in_channels[1] + base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.norm_2a = nn.LayerNorm(base_channels)

        self.up_1a = nn.Sequential(
            nn.Conv2d(in_channels[0] + base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.norm_1a = nn.LayerNorm(base_channels)

        self.up_0a = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.norm_0a = nn.LayerNorm(base_channels)

        self.conv = nn.Conv2d(base_channels, n_classes, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def norm(self, x, n):
        x = x.permute(0, 2, 3, 1)
        x = n(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, inputs):
        for i in range(len(inputs)):
            inputs[i] = inputs[i].permute(0, 3, 1, 2)
        
        up_4a = self.up_4a(inputs[3])
        up_4a = self.norm(up_4a, self.norm_4a)
        up_3a = self.up_3a(torch.cat((up_4a, inputs[2]), dim=1))
        up_3a = self.norm(up_3a, self.norm_3a)
        up_2a = self.up_2a(torch.cat((up_3a, inputs[1]), dim=1))
        up_2a = self.norm(up_2a, self.norm_2a)
        up_1a = self.up_1a(torch.cat((up_2a, inputs[0]), dim=1))
        up_1a = self.norm(up_1a, self.norm_1a)
        up_0a = self.up_0a(up_1a)
        up_0a = self.norm(up_0a, self.norm_0a)
        out = self.conv(up_0a)
        out = self.softmax(out)

        return out

# class LinearUpsample(nn.Module):
#     def __init__(self, in_channels=[], n_classes=20, base_channels=96):
#         super().__init__()

#         self.reduce_1 = nn.Sequential(
#             nn.Linear(in_channels[3], base_channels),
#             nn.LeakyReLU(),
#             nn.LayerNorm(base_channels)
#         )

#         self.reduce_2 = nn.Sequential(
#             nn.Linear(in_channels[2] + base_channels, base_channels),
#             nn.LeakyReLU(),
#             nn.LayerNorm(base_channels)
#         )

#         self.reduce_3 = nn.Sequential(
#             nn.Linear(in_channels[1] + base_channels, base_channels),
#             nn.LeakyReLU(),
#             nn.LayerNorm(base_channels)
#         )

#         self.reduce_4 = nn.Sequential(
#             nn.Linear(in_channels[0] + base_channels, base_channels),
#             nn.LeakyReLU(),
#             nn.LayerNorm(base_channels)
#         )

#         self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
#         self.cls = nn.Sequential(
#             nn.Linear(base_channels, 4 * base_channels),
#             nn.LeakyReLU(),
#             nn.LayerNorm(4 * base_channels),
#             nn.Linear(4 * base_channels, n_classes)
#         )

#         self.softmax = nn.Softmax(dim=3)

#     def up(self, x):
#         x = x.permute(0, 3, 1, 2)
#         x = self.upsample(x)
#         x = x.permute(0, 2, 3, 1)
#         return x

#     def forward(self, xs):
#         x_4, x_3, x_2, x_1 = xs

#         x_1 = self.reduce_1(x_1)
#         x_2 = self.reduce_2(torch.cat([x_2, self.up(x_1)], dim=3))
#         x_3 = self.reduce_3(torch.cat([x_3, self.up(x_2)], dim=3))
#         x_4 = self.reduce_4(torch.cat([x_4, self.up(x_3)], dim=3))

#         out = self.up(self.up(x_4))
#         out = self.cls(out)
#         out = self.softmax(out)

#         return out



class SwinFusion(SwinTransformer):
    def __init__(self):
        super().__init__(in_channels=5, patch_size=[4, 4], embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=[8, 8])
        self.fusion_1 = SwinBasedFusion(in_dims=192, out_dims=96)
        self.fusion_2 = SwinBasedFusion(in_dims=384, out_dims=192)
        self.fusion_3 = SwinBasedFusion(in_dims=768, out_dims=384)
        self.fusion_4 = SwinBasedFusion(in_dims=1536, out_dims=768)

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

class FusionNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = SwinBackbone(backbone)

        self.fusion = SwinFusion()

        self.upsample = ConvUpsample(in_channels=[96, 192, 384, 768], n_classes=20, base_channels=96)

    def forward(self, pcd_feature, img_feature):
        img_features = self.backbone(img_feature)

        pcd_out = self.fusion(pcd_feature, img_features)

        img_out = self.upsample(img_features)

        return pcd_out, img_out
        
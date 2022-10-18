import torch
from torch import nn
import torch.nn.functional as F
from .pmf_net import RGBDecoder, ResNet
from .pointtransformer import PointTransformerSeg, PointTransformerBlock

class FusionBlock(nn.Module):
    def __init__(self, pcd_features, img_features):
        super(FusionBlock, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(pcd_features+img_features, pcd_features, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(pcd_features)
        )

    def forward(self, pcd, img):
        cat = torch.cat((pcd, img), dim=1)
        return self.fuse(cat)


class Fusion(nn.Module):
    def __init__(self, nclasses=20, img_features=[64,128,256,512]):
        super(Fusion, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(nclasses, img_features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(img_features[0]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(img_features[0], img_features[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(img_features[1]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(img_features[1], img_features[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(img_features[2]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(img_features[2], img_features[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(img_features[3]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fusion1 = FusionBlock(64, img_features[0])
        self.fusion2 = FusionBlock(128, img_features[1])
        self.fusion3 = FusionBlock(256, img_features[2])
        self.fusion4 = FusionBlock(512, img_features[3])

        self.up4 = nn.Sequential(
            nn.Conv2d(img_features[3], img_features[2], 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(img_features[2]),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(img_features[2], img_features[1], 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(img_features[1]),
            nn.Upsample(scale_factor=2, mode="bilinear")

        )

        self.up2 = nn.Sequential(
            nn.Conv2d(img_features[1], img_features[0], 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(img_features[0]),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )

        self.conv = nn.Conv2d(32, nclasses, kernel_size=3, padding=1)

    def forward(self, lidar_image, img_features):
        out1 = self.conv1(lidar_image)

        fuse1 = self.fusion1(img_features[0], out1)
        out2 = self.conv2(fuse1)

        fuse2 = self.fusion2(img_features[1], out2)
        out3 = self.conv3(fuse2)

        fuse3 = self.fusion3(img_features[2], out3)
        out4 = self.conv4(fuse3)

        fuse4 = self.fusion4(img_features[3], out4)

        up4 = self.up4(fuse4)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)

        out = self.conv(up1)
        logits = F.softmax(out, dim=1)

        return logits

class ResPoint(nn.Module):
    def __init__(self, pcd_channels=5, img_channels=3, nclasses=20, 
                 imagenet_pretrained=True, image_backbone="resnet34"):
        super(ResPoint, self).__init__()

        self.nclasses = nclasses

        self.camera_stream_encoder = ResNet(
            in_channels=img_channels,
            pretrained=imagenet_pretrained,
            backbone=image_backbone
        )

        self.camera_stream_decoder = RGBDecoder(
            self.camera_stream_encoder.feature_channels,
            nclasses=nclasses,
            base_channels=self.camera_stream_encoder.expansion*16
        )

        self.point_transformer = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], c=pcd_channels, k=nclasses)

        self.fusion = Fusion(nclasses=nclasses)

    def forward(self, pcd_features, img_features, label_mask):
        img_features = self.camera_stream_encoder(img_features)
        
        # to spremen da bo delalo na predprocesorju
        pxo = pcd_features.permute(0, 2, 3, 1)
        pxo = pxo[label_mask]

        xyz = pxo[:, :3]
        feat = pxo[:, 3:]
        offset = torch.tensor([xyz.shape[0]], dtype=torch.int32, device=xyz.device)

        lidar_features = self.point_transformer([xyz.contiguous(), feat.contiguous(), offset])

        lidar_image = torch.zeros((*label_mask.shape, self.nclasses), dtype=pcd_features.dtype, device=pcd_features.device)
        lidar_image[label_mask] = lidar_features
        lidar_image = lidar_image.permute(0, 3, 1, 2)

        lidar_pred = self.fusion(lidar_image, img_features)
        camera_pred = self.camera_stream_decoder(img_features)

        return lidar_pred, camera_pred




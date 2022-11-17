import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .pmf_net import  ResNet

from lib.pointops.functions import pointops
from pc_processor.models.pointtransformer import PointTransformerBlock

# Modified from pointtransformer.PointTransformerBlock https://github.com/POSTECH-CVLab/point-transformer
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, share_channels=8, nsample=8):
        super(SelfAttentionBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.transformer2 = SelfAttentionLayer(out_channels, out_channels, share_channels, nsample)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.linear3 = nn.Linear(out_channels, out_channels, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]

# Modified from pointtransformer.PointTransformerLayer
class SelfAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, share_channels=8, nsample=8):
        super(SelfAttentionLayer, self).__init__()
        self.nsample = nsample
        self.mid_channels = out_channels
        self.share_channels = share_channels
        self.out_channels = out_channels
        self.linear_q = nn.Linear(in_channels, out_channels)
        self.linear_k = nn.Linear(in_channels, out_channels)
        self.linear_v = nn.Linear(in_channels, out_channels)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_channels))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
                                    nn.Linear(out_channels, out_channels // share_channels),
                                    nn.BatchNorm1d(out_channels // share_channels), nn.ReLU(inplace=True),
                                    nn.Linear(out_channels // share_channels, out_channels // share_channels))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        p, x, o = pxo

        # print("p.shape", p.shape)
        # print("x.shape", x.shape)
        # print("o.shape", o.shape)
        # print()

        x_query = self.linear_q(x)
        x_key = self.linear_k(x)
        x_value = self.linear_v(x)

        # print("x_query.shape", x_query.shape)
        # print("x_key.shape", x_key.shape)
        # print("x_value.shape", x_value.shape)
        # print()

        idx, _ = pointops.knnquery(self.nsample, p, p, o, o)
        x_key = pointops.queryandgroup(self.nsample, p, p, x_key, idx, o, o, use_xyz=True)
        x_value = pointops.queryandgroup(self.nsample, p, p, x_value, idx, o, o, use_xyz=False)

        # print("x_key.shape", x_key.shape)
        # print("x_value.shape", x_value.shape)
        # print()

        p_r = x_key[:, :, 0:3]
        x_key = x_key[:, :, 3:]

        # print("p_r.shape", p_r.shape)
        # print("x_key.shape", x_key.shape)
        # print()

        # Positional encoding
        for i, layer in enumerate(self.linear_p):
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        
        # print("p_r.shape", p_r.shape)
        # print()

        # Calculate the attention weights
        w = x_key - x_query.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_channels // self.mid_channels, self.mid_channels).sum(2)  # (n, nsample, c)

        # print("w.shape", w.shape)
        # print()

        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)

        # print("w.shape", w.shape)
        # print()

        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_value.shape
        s = self.share_channels

        x = ((x_value + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)

        # print((x_value + p_r).view(n, nsample, s, c // s).shape)
        # print(w.unsqueeze(2).shape)
        # print(((x_value + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).shape)
        # print(((x_value + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).shape)

        # print("x.shape", x.shape)
        # print()

        #exit()

        return x

# Modified from pointtransformer.PointTransformerBlock https://github.com/POSTECH-CVLab/point-transformer
class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, share_channels=8, nsample=8):
        super(CrossAttentionBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels1, out_channels, bias=False)
        self.linear2 = nn.Linear(in_channels2, out_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.transformer2 = CrossAttentionLayer(out_channels, out_channels, share_channels, nsample)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.linear3 = nn.Linear(out_channels, out_channels, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo1, pxo2): # pxo1: query, pxo2: key and value
        p1, x1, o1 = pxo1
        p2, x2, o2 = pxo2

        x1 = self.relu(self.bn1(self.linear1(x1)))
        x2 = self.relu(self.bn2(self.linear2(x2)))
        identity = x1

        x = self.relu(self.bn2(self.transformer2([p1, x1, o1], [p2, x2, o2])))

        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p1, x, o1]

# Modified from pointtransformer.PointTransformerLayer
class CrossAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, share_channels=8, nsample=8):
        super(CrossAttentionLayer, self).__init__()
        self.nsample = nsample
        self.mid_channels = out_channels
        self.share_channels = share_channels
        self.out_channels = out_channels
        self.linear_q = nn.Linear(in_channels, out_channels)
        self.linear_k = nn.Linear(in_channels, out_channels)
        self.linear_v = nn.Linear(in_channels, out_channels)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_channels))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
                                    nn.Linear(out_channels, out_channels // share_channels),
                                    nn.BatchNorm1d(out_channels // share_channels), nn.ReLU(inplace=True),
                                    nn.Linear(out_channels // share_channels, out_channels // share_channels))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo1, pxo2): # pxo1: query, pxo2: key and value
        p1, x1, o1 = pxo1
        p2, x2, o2 = pxo2

        x_query = self.linear_q(x1)
        x_key = self.linear_k(x2)
        x_value = self.linear_v(x2)

        idx, _ = pointops.knnquery(self.nsample, p2, p1, o2, o1)
        x_key = pointops.queryandgroup(self.nsample, p2, p1, x_key, idx, o2, o1, use_xyz=True)
        x_value = pointops.queryandgroup(self.nsample, p2, p1, x_value, idx, o2, o1, use_xyz=False)

        p_r = x_key[:, :, 0:3]
        x_key = x_key[:, :, 3:]

        # Positional encoding
        for i, layer in enumerate(self.linear_p):
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        
        # Calculate the attention weights
        w = x_key - x_query.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_channels // self.mid_channels, self.mid_channels).sum(2)  # (n, nsample, c)

        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)

        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_value.shape
        s = self.share_channels

        x = ((x_value + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)

        return x

class LinearWrapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearWrapper, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return [p, x, o]

class TransFusion(nn.Module):
    def __init__(self, img_channels=3, pcd_features=5, nclasses=20, imagenet_pretrained=True, image_backbone="resnet34"):
        super(TransFusion, self).__init__()

        self.nclasses = nclasses

        self.camera_encoder = ResNet(
            in_channels=img_channels,
            pretrained=imagenet_pretrained,
            backbone=image_backbone
        )

        self.pcd_only_attention = self.self_attention_blocks(num_blocks=3, in_channels=pcd_features, out_channels=32)

        self.cross_attention_1 = CrossAttentionBlock(in_channels1=64, in_channels2=32, out_channels=32)
        self.self_attention_1 = self.self_attention_blocks(num_blocks=2, in_channels=32, out_channels=32)

        self.cross_attention_2 = CrossAttentionBlock(in_channels1=32, in_channels2=128, out_channels=32)
        self.self_attention_2 = self.self_attention_blocks(num_blocks=2, in_channels=32, out_channels=32)

        self.cross_attention_3 = CrossAttentionBlock(in_channels1=32, in_channels2=256, out_channels=32)
        self.self_attention_3 = self.self_attention_blocks(num_blocks=2, in_channels=32, out_channels=32)

        self.cross_attention_4 = CrossAttentionBlock(in_channels1=32, in_channels2=512, out_channels=32)
        self.self_attention_4 = self.self_attention_blocks(num_blocks=2, in_channels=32, out_channels=32)

        self.cross_attention_5 = CrossAttentionBlock(in_channels1=3, in_channels2=32, out_channels=32)
        self.self_attention_5 = self.self_attention_blocks(num_blocks=2, in_channels=32, out_channels=32)

        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, nclasses)
        )

        print("MODEL INITIALIZED")

    def self_attention_blocks(self, num_blocks, in_channels, out_channels):
        layers = []
        layers.append(LinearWrapper(in_channels, out_channels))
        for _ in range(num_blocks):
            layers.append(SelfAttentionBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    # Spremeni da bo delalo z batchi
    def image_to_points(self, image, scale):
        image = image.permute(0, 2, 3, 1)[0]
        py, px = np.mgrid[0:image.shape[0], 0:image.shape[1]]

        coords = np.stack([py, px], axis=-1).reshape(-1, 2)
        coords *= scale
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1))], axis=-1)

        image = image.reshape(-1, image.shape[-1])

        return torch.Tensor(coords).cuda(), torch.Tensor(image).cuda(), torch.IntTensor([image.shape[0]]).cuda()

    
    def forward(self, image, pointcloud):
        image_features = self.camera_encoder(image)

        # Spremen da bo delalo z batchi!!!
        pcp_img = pointcloud[0, :, :2].contiguous().float()
        pcp_img = torch.cat([pcp_img, torch.zeros(pcp_img.shape[0], 1).cuda()], dim=1)

        pcp_world = pointcloud[0, :, 2:5].contiguous().float()
        pc_x = pointcloud[0, :, 2:].contiguous().float()
        pc_o = torch.cuda.IntTensor([pcp_img.shape[0]])

        pcp_world, pc_x, pc_o = self.pcd_only_attention([pcp_world, pc_x, pc_o])

        # prvič vsak pixel slike attenda na točke, tko da dobimo točko za vsak pixel
        p_img, x_img, o_img = self.image_to_points(image_features[0], 2)
        p, x, o = self.cross_attention_1([p_img, x_img, o_img], [pcp_img, pc_x, pc_o])
        p, x, o = self.self_attention_1([p, x, o])

        # # od tuki naprej pa vsaka točke attenda na pixel
        p_img, x_img, o_img = self.image_to_points(image_features[1], 4)
        p, x, o = self.cross_attention_2([p, x, o], [p_img, x_img, o_img])
        p, x, o = self.self_attention_2([p, x, o])

        p_img, x_img, o_img = self.image_to_points(image_features[2], 8)
        p, x, o = self.cross_attention_3([p, x, o], [p_img, x_img, o_img])
        p, x, o = self.self_attention_3([p, x, o])

        p_img, x_img, o_img = self.image_to_points(image_features[3], 16)
        p, x, o = self.cross_attention_4([p, x, o], [p_img, x_img, o_img])
        p, x, o = self.self_attention_4([p, x, o])

        # p_img, x_img, o_img = self.image_to_points(image, 1)
        # p, x, o = self.cross_attention_5([p_img, x_img, o_img], [p, x, o])
        # p, x, o = self.self_attention_5([p, x, o])

        coords, _, o_new = self.image_to_points(image, 1)
        x = pointops.interpolation(p, coords, x, o, o_new)

        x = self.classifier(x)

        image = x.reshape(image.shape[0], image.shape[2], image.shape[3], -1)

        return image
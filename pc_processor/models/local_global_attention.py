import torch
from torch import nn
from .pmf_net import ResNet

# Global Multi Head Self Attention, adapted from https://github.com/makeesyai/makeesy-deep-learning/blob/main/self_attention/multiheaded_attention_optimized.py
# Computes attention globally across the image, using patches as "words"
class GlobalMHSA(nn.Module):
    def __init__(self, in_features, heads_dim, num_heads=2):
        super(GlobalMHSA, self).__init__()

        self.num_heads = num_heads
        self.heads_dim = heads_dim

        self.linear_q = nn.Linear(in_features, heads_dim * num_heads, bias=False)
        self.linear_k = nn.Linear(in_features, heads_dim * num_heads, bias=False)
        self.linear_v = nn.Linear(in_features, heads_dim * num_heads, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.linear_final = nn.Linear(heads_dim * num_heads, in_features, bias=False)

    def forward(self, inputs):
        batches, patches, _ = inputs.shape

        q = self.linear_q(inputs).view(batches, patches, self.num_heads, self.heads_dim).transpose(1, 2)
        k = self.linear_k(inputs).view(batches, patches, self.num_heads, self.heads_dim).transpose(1, 2)
        v = self.linear_v(inputs).view(batches, patches, self.num_heads, self.heads_dim).transpose(1, 2)

        attention = self.softmax(q @ k.transpose(-2, -1) / (self.heads_dim ** (1 / 2)))

        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(batches, patches, self.num_heads * self.heads_dim)

        return self.linear_final(out)

# Local Multi Head Self Attention
# Computes attention locally for each patch, using every pixel as a "word"
class LocalMHSA(GlobalMHSA):
    def __init__(self, in_features, heads_dim, num_heads=2):
        super(LocalMHSA, self).__init__(in_features, heads_dim, num_heads=num_heads)

    def forward(self, inputs1, inputs2):
        # It differs only in dimensions
        batches1, patches1, pixels1, _ = inputs1.shape
        batches2, patches2, pixels2, _ = inputs2.shape

        q = self.linear_q(inputs1).view(batches1, patches1, pixels1, self.num_heads, self.heads_dim).transpose(2, 3)
        k = self.linear_k(inputs2).view(batches2, patches2, pixels2, self.num_heads, self.heads_dim).transpose(2, 3)
        v = self.linear_v(inputs2).view(batches2, patches2, pixels2, self.num_heads, self.heads_dim).transpose(2, 3)

        attention = self.softmax(q @ k.transpose(-2, -1) / (self.heads_dim ** (1 / 2)))

        out = attention @ v
        out = out.transpose(2, 3).contiguous().view(batches1, patches1, pixels1, self.num_heads * self.heads_dim)

        return self.linear_final(out)

class TransformerBlock(nn.Module):
    def __init__(self, in_features, heads_dim, num_heads, mlp_ratio, attention_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_features)
        self.mhsa = attention_type(in_features=in_features,
                                   heads_dim=heads_dim, 
                                   num_heads=num_heads)
        self.norm2 = nn.LayerNorm(in_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features * mlp_ratio),
            nn.GELU(),
            nn.Linear(in_features * mlp_ratio, in_features)
        )

    def forward(self, inputs1, inputs2=None):
        if inputs2 is None:
            out = inputs1 + self.mhsa(self.norm1(inputs1))
        else:
            out = inputs1 + self.mhsa(self.norm1(inputs1), self.norm1(inputs2))
        out = out + self.mlp(self.norm2(out))

        return out

# Can be used as a cross attention layer
class LocalTransformerLayer(nn.Module):
    def __init__(self, in_features, hidden_d, heads_dim, num_heads, num_pixels_1, num_pixels_2=None, mlp_ratio=4):
        super(LocalTransformerLayer, self).__init__()
        
        self.linear_mapper = nn.Linear(in_features, hidden_d)

        self.positional_embeding_1 = torch.zeros((1, num_pixels_1, hidden_d)).to("cuda")
        self.positional_embeding_1[:, :, 0::2] = torch.sin(torch.arange(0, num_pixels_1).unsqueeze(1) / (10000 ** (torch.arange(0, hidden_d, 2) / hidden_d)))
        self.positional_embeding_1[:, :, 1::2] = torch.cos(torch.arange(0, num_pixels_1).unsqueeze(1) / (10000 ** (torch.arange(1, hidden_d, 2) / hidden_d)))

        if num_pixels_2:
            self.positional_embeding_2 = torch.zeros((1, num_pixels_2, hidden_d)).to("cuda")
            self.positional_embeding_2[:, :, 0::2] = torch.sin(torch.arange(0, num_pixels_2).unsqueeze(1) / (10000 ** (torch.arange(0, hidden_d, 2) / hidden_d)))
            self.positional_embeding_2[:, :, 1::2] = torch.cos(torch.arange(0, num_pixels_2).unsqueeze(1) / (10000 ** (torch.arange(1, hidden_d, 2) / hidden_d)))

        self.transformer_block = TransformerBlock(in_features=hidden_d,
                                                  heads_dim=heads_dim,
                                                  num_heads=num_heads,
                                                  mlp_ratio=mlp_ratio,
                                                  attention_type=LocalMHSA)

    def forward(self, inputs1, inputs2=None):
        tokens1 = self.linear_mapper(inputs1)
        pos_embed1 = tokens1 + self.positional_embeding_1

        if inputs2 is not None:
            tokens2 = self.linear_mapper(inputs2)
            pos_embed2 = tokens2 + self.positional_embeding_2

            return self.transformer_block(pos_embed1, pos_embed2)
        
        return self.transformer_block(pos_embed1, pos_embed1)


class GlobalTransformerLayer(nn.Module):
    def __init__(self, in_features, hidden_d, heads_dim, num_heads, mlp_ratio=4):
        super(GlobalTransformerLayer, self).__init__()

        self.hidden_d = hidden_d

        self.linear_mapper = nn.Linear(in_features, hidden_d)

        self.transformer_block = TransformerBlock(in_features=hidden_d,
                                                  heads_dim=heads_dim,
                                                  num_heads=num_heads,
                                                  mlp_ratio=mlp_ratio,
                                                  attention_type=GlobalMHSA)

    def forward(self, input):
        _, num_patches, _ = input.shape

        tokens = self.linear_mapper(input)
        pos_embed = torch.zeros((1, num_patches, self.hidden_d)).to(input.device)
        pos_embed[:, :, 0::2] = torch.sin(torch.arange(0, num_patches).unsqueeze(1) / (10000 ** (torch.arange(0, self.hidden_d, 2) / self.hidden_d)))
        pos_embed[:, :, 1::2] = torch.cos(torch.arange(0, num_patches).unsqueeze(1) / (10000 ** (torch.arange(1, self.hidden_d, 2) / self.hidden_d)))

        pos_embed_tokens = tokens + pos_embed

        return self.transformer_block(pos_embed_tokens)


class LidarRGBFusion(nn.Module):
    def __init__(self, img_channels=3, pcd_channels=5, nclasses=20, imagenet_pretrained=True, image_backbone="resnet34"):
        super(LidarRGBFusion, self).__init__()

        self.camera_encoder = ResNet(
            in_channels=img_channels,
            pretrained=imagenet_pretrained,
            backbone=image_backbone
        )

        self.pcd_channels = pcd_channels
        self.nclasses = nclasses

        self.lidar_patch_size = 8
        self.img_patch_sizes = [4, 2, 1, 1]

        #self.global_1 = GlobalTransformerLayer(in_features=64, hidden_d=64, heads_dim=64, num_heads=8)
        self.local_1 = LocalTransformerLayer(in_features=5, hidden_d=32, heads_dim=32, num_heads=1, num_pixels_1=self.lidar_patch_size**2)

        self.linear_1 = nn.Linear(64, 32)
        self.cross_1 = LocalTransformerLayer(in_features=32, hidden_d=32, heads_dim=32, num_heads=1, num_pixels_1=self.lidar_patch_size**2, num_pixels_2=16)

        #self.global_2 = GlobalTransformerLayer(in_features=32, hidden_d=32, heads_dim=32, num_heads=8)
        self.local_2 = LocalTransformerLayer(in_features=32, hidden_d=32, heads_dim=32, num_heads=1, num_pixels_1=self.lidar_patch_size**2)

        self.linear_2 = nn.Linear(128, 32)
        self.cross_2 = LocalTransformerLayer(in_features=32, hidden_d=32, heads_dim=32, num_heads=1, num_pixels_1=self.lidar_patch_size**2, num_pixels_2=4)

        #self.global_3 = GlobalTransformerLayer(in_features=32, hidden_d=32, heads_dim=32, num_heads=8)
        self.local_3 = LocalTransformerLayer(in_features=32, hidden_d=32, heads_dim=32, num_heads=1, num_pixels_1=self.lidar_patch_size**2)
        
        self.linear_3 = nn.Linear(256, 32)
        self.cross_3 = LocalTransformerLayer(in_features=32, hidden_d=32, heads_dim=32, num_heads=1, num_pixels_1=self.lidar_patch_size**2, num_pixels_2=1)

        #self.global_4 = GlobalTransformerLayer(in_features=32, hidden_d=32, heads_dim=32, num_heads=8)
        self.local_4 = LocalTransformerLayer(in_features=32, hidden_d=32, heads_dim=32, num_heads=1, num_pixels_1=self.lidar_patch_size**2)

        # self.cross_4 = LocalTransformerLayer(in_features=32, hidden_d=32, heads_dim=32, num_heads=1, num_pixels_1=self.lidar_patch_size**2, num_pixels_2=1)

        # #self.global_5 = GlobalTransformerLayer(in_features=32, hidden_d=32, heads_dim=32, num_heads=8)
        # self.local_5 = LocalTransformerLayer(in_features=32, hidden_d=32, heads_dim=32, num_heads=1, num_pixels_1=self.lidar_patch_size**2)

        self.cls = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, nclasses)
        )

    def forward(self, img, pcd):
        import time
        img_features = self.camera_encoder(img)

        # Patchify pcd into 16x16 patches
        n = self.lidar_patch_size
        local_patches = pcd.unfold(2, n, n).unfold(3, n, n)
        unfold_shape = local_patches.shape

        local_patches = local_patches.reshape(1, self.pcd_channels, -1, n, n).flatten(3).permute(0,2,3,1)
        # print(local_patches.shape)
        local_patches = self.local_1(local_patches)
        
        n1 = self.img_patch_sizes[0]
        img_patches = img_features[0].unfold(2, n1, n1).unfold(3, n1, n1).reshape(1, 64, -1, n1, n1).flatten(3).permute(0,2,3,1)
        img_patches = self.linear_1(img_patches)
        local_patches = self.cross_1(local_patches, img_patches)
        local_patches = self.local_2(local_patches)

        n2 = self.img_patch_sizes[1]
        img_patches = img_features[1].unfold(2, n2, n2).unfold(3, n2, n2).reshape(1, 128, -1, n2, n2).flatten(3).permute(0,2,3,1)
        img_patches = self.linear_2(img_patches)
        local_patches = self.cross_2(local_patches, img_patches)
        local_patches = self.local_3(local_patches)

        n3 = self.img_patch_sizes[2]
        img_patches = img_features[2].unfold(2, n3, n3).unfold(3, n3, n3).reshape(1, 256, -1, n3, n3).flatten(3).permute(0,2,3,1)
        img_patches = self.linear_3(img_patches)
        local_patches = self.cross_3(local_patches, img_patches)
        local_patches = self.local_4(local_patches)

        local_patches = self.cls(local_patches)

        batches, channels, patch1, patch2, w, h = unfold_shape
        output = local_patches.permute(0,3,1,2).contiguous()
        output = output.view(batches, self.nclasses, patch1, patch2, w, h)
        output = output.permute(0, 1, 2, 4, 3, 5).contiguous()
        output = output.view(batches, self.nclasses, patch1*w, patch2*h)

        # import matplotlib.pyplot as plt
        # plt.imshow(output[0].argmax(0).detach().cpu().numpy())
        # plt.show()

        return output
    





        





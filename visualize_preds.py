import pc_processor
import numpy as np
import torch
import yaml

config = yaml.safe_load(open("tasks/pmf/config_server_kitti.yaml", "r"))

valset = pc_processor.dataset.semantic_kitti.SemanticKitti(
    root="../datasets/broken-kitti/sequences/",
    sequences=[9], #sequences=[0,1,2,3,4,5,6,7,9,10],
    config_path="pc_processor/dataset/semantic_kitti/semantic-kitti.yaml",
    has_label = True,
    has_image=True
)

val_pv_loader = pc_processor.dataset.PerspectiveViewLoader(
    dataset=valset,
    config=config,
    is_train=False,
    return_uproj=False
)

val_loader = torch.utils.data.DataLoader(
    val_pv_loader,
    batch_size=1,
    num_workers=4,
    shuffle=False,
    drop_last=False
)

best_pmf_model_path = "/home/matej/diploma/Diploma/experiments/PMF-SemanticKitti/log_SemanticKitti_PMFNet-resnet34_bs1-lr0.001_baseline_timestamp/checkpoint/best_IOU_model.pth"
pmf = pc_processor.models.PMFNet(
    pcd_channels=5,
    img_channels=3,
    nclasses=20,
    base_channels=32,
    image_backbone="resnet34",
    imagenet_pretrained=True
)
state_dict = torch.load(best_pmf_model_path, map_location="cpu")
pmf.load_state_dict(state_dict)
pmf.cuda()
pmf.eval()
print("Loaded PMF model")

best_respoint_model_path = "/home/matej/diploma/Diploma/experiments/PMF-SemanticKitti/log_SemanticKitti_PMFNet-resnet34_bs1-lr0.001_point_transformer/checkpoint/best_IOU_model.pth"
respoint =  pc_processor.models.ResPoint(
    pcd_channels=5,
    img_channels=3,
    nclasses=20,
    image_backbone="resnet34",
    imagenet_pretrained=True
)
state_dict = torch.load(best_respoint_model_path, map_location="cpu")
respoint.load_state_dict(state_dict)
respoint.cuda()
respoint.eval()
print("LOADED RESPOINT")

best_transformer = "/home/matej/diploma/Diploma/experiments/PMF-SemanticKitti/log_SemanticKitti_PMFNet-resnet34_bs1-lr0.001_point_transformer/checkpoint/best_IOU_model_transformer.pth"
transformer = pc_processor.models.pointtransformer_seg_repro(pcd_channels=5, num_classes=20)
state_dict = torch.load(best_transformer, map_location="cpu")
transformer.load_state_dict(state_dict)
transformer.cuda()
transformer.eval()
print("LOADED TRANSFORMER")


import matplotlib.pyplot as plt

with torch.inference_mode():
    feature_mean = torch.Tensor(config["sensor"]["img_mean"]).unsqueeze(0).unsqueeze(2).unsqueeze(2).cuda()
    feature_std = torch.Tensor(config["sensor"]["img_stds"]).unsqueeze(0).unsqueeze(2).unsqueeze(2).cuda()
    # for i, (input_feature, input_mask, input_label, uproj_x_idx, uproj_y_idx, uproj_depth) in enumerate(val_loader):
    #     uproj_x_idx = uproj_x_idx[0].long().cuda()
    #     uproj_y_idx = uproj_y_idx[0].long().cuda()
    #     uproj_depth = uproj_depth[0].cuda()

    #     input_feature = input_feature.cuda()
    #     proj_depth = input_feature[0, 0, ...].clone()
    #     proj_depth = proj_depth - proj_depth.eq(0).float()
    #     # padding
    #     h_pad = config["sensor"]["h_pad"]
    #     w_pad = config["sensor"]["w_pad"]
    #     padding_layer = torch.nn.ZeroPad2d(
    #         (w_pad, w_pad, h_pad, h_pad))

    #     input_feature = padding_layer(input_feature)
    #     input_mask = input_mask.cuda()
    #     input_mask = padding_layer(input_mask)
    #     input_feature[:, 0:5] = (
    #         input_feature[:, 0:5] - feature_mean) / feature_std * \
    #         input_mask.unsqueeze(1).expand_as(input_feature[:, 0:5])
    #     pcd_feature = input_feature[:, 0:5]
    #     img_feature = input_feature[:, 5:8]

    #     input_label = input_label.long().cuda()
    
    for i, (input_feature, input_mask, input_label) in enumerate(val_loader):
        input_feature = input_feature.cuda()
        input_mask = input_mask.cuda()
        input_feature[:, 0:5] = (
            input_feature[:, 0:5] - feature_mean) / feature_std * \
            input_mask.unsqueeze(1).expand_as(input_feature[:, 0:5])
        pcd_feature = input_feature[:, 0:5]
        img_feature = input_feature[:, 5:8]
        input_label = input_label.cuda().long()
        label_mask = input_label.gt(0)

        # to spremen da bo delalo na predprocesorju
        pxo = pcd_feature.permute(0, 2, 3, 1)
        pxo = pxo[label_mask]
        target = input_label[label_mask]

        if target.shape[-1] == 1:
            target = target[:, 0]

        xyz = pxo[:, :3].contiguous()
        feat = pxo[:, 3:].contiguous()
        offset = torch.tensor([xyz.shape[0]], dtype=torch.int32, device=xyz.device)

        pmf_lidar, pmf_image = pmf(pcd_feature, img_feature)

        encoded_lidar = transformer([xyz, feat, offset])

        respoint_lidar, respoint_image = respoint(encoded_lidar, img_feature, label_mask)

        plt.subplot(2,2,1)
        plt.imshow(pmf_lidar[0].argmax(0).cpu().numpy())
        plt.title("PMF Lidar")
        plt.axis(False)

        plt.subplot(2,2,2)
        plt.imshow(pmf_image[0].argmax(0).cpu().numpy())
        plt.title("PMF Image")
        plt.axis(False)

        plt.subplot(2,2,3)
        plt.imshow(respoint_lidar[0].argmax(0).cpu().numpy())
        plt.title("ResPoint Lidar")
        plt.axis(False)

        plt.subplot(2,2,4)
        plt.imshow(respoint_image[0].argmax(0).cpu().numpy())
        plt.title("ResPoint Image")
        plt.axis(False)
        plt.show()
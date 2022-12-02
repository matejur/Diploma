import pc_processor
import numpy as np
import torch
import yaml

torch.manual_seed(0)
torch.cuda.manual_seed(0)

config = yaml.safe_load(open("tasks/pmf/config_server_kitti.yaml", "r"))

valset = pc_processor.dataset.semantic_kitti.SemanticKitti(
    root="../datasets/semantic-kitti/sequences/",
    sequences=[0,1,2,8], #sequences=[0,1,2,3,4,5,6,7,9,10],
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
    shuffle=True,
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

best_respoint_model_path = "/home/matej/diploma/Diploma/experiments/PMF-SemanticKitti/log_SemanticKitti_PMFNet-resnet34_bs1-lr0.001_patches_transformer_32_1head/checkpoint/best_IOU_model.pth"
myModel =  pc_processor.models.LidarRGBFusion()
state_dict = torch.load(best_respoint_model_path, map_location="cpu")
myModel.load_state_dict(state_dict)
myModel.cuda()
myModel.eval()
print("LOADED RESPOINT")

import matplotlib.pyplot as plt

with torch.inference_mode():
    feature_mean = torch.Tensor(config["sensor"]["img_mean"]).unsqueeze(
        0).unsqueeze(2).unsqueeze(2).cuda()
    feature_std = torch.Tensor(config["sensor"]["img_stds"]).unsqueeze(
        0).unsqueeze(2).unsqueeze(2).cuda()

    for i, (input_feature, input_mask, input_label) in enumerate(val_loader):
        # t_process_start = time.time()
        input_feature = input_feature.cuda()
        input_mask = input_mask.cuda()
        input_feature[:, 0:5] = (
            input_feature[:, 0:5] - feature_mean) / feature_std * \
            input_mask.unsqueeze(1).expand_as(input_feature[:, 0:5])

        pcd_feature = input_feature[:, 0:5]
        img_feature = input_feature[:, 5:8]
        input_label = input_label.cuda().long()
        label_mask = input_label.gt(0)

        mine = myModel(img_feature, pcd_feature)

        theirs, _ = pmf(pcd_feature, img_feature)

        plt.subplot(3,1,1)
        plt.imshow(mine[0].argmax(0).cpu().detach().numpy())
        plt.title("Mine")

        plt.subplot(3,1,2)
        plt.imshow(theirs[0].argmax(0).cpu().detach().numpy())
        plt.title("Theirs")

        plt.subplot(3,1,3)
        plt.imshow(img_feature[0].permute(1,2,0).cpu().detach().numpy())
        plt.show()
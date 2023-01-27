import pc_processor
import numpy as np
import torch
import yaml

config = yaml.safe_load(open("tasks/swin/config_server_kitti.yaml", "r"))
color_config = yaml.safe_load(open("pc_processor/dataset/semantic_kitti/semantic-kitti.yaml", "r"))

valset = pc_processor.dataset.semantic_kitti.SemanticKitti(
    #root="../datasets/broken-kitti/sequences/",
    #sequences=[9],
    root="../datasets/semantic-kitti-fov/sequences/",
    sequences=[8],
    #sequences=[0,1,2,3,4,5,6,7,8,9,10],
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

best_pmf_model_path = "/home/matej/diploma/Diploma/experiments/PMF-SemanticKitti/log_SemanticKitti_PMFNet-resnet34_bs1-lr0.001_pmf/checkpoint/best_IOU_model.pth"
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

best_swin_model = "/home/matej/diploma/Diploma/experiments/PMF-SemanticKitti/log_SemanticKitti_PMFNet-swin_tiny_bs2-lr0.0001_swin_no_dropout_batch_8/checkpoint/best_IOU_model.pth"
swin_model_pcd_q =  pc_processor.models.FusionCrossNet(backbone="swin_tiny", pcd_q=True, size=7)
state_dict = torch.load(best_swin_model, map_location="cpu")
swin_model_pcd_q.load_state_dict(state_dict)
swin_model_pcd_q.cuda()
swin_model_pcd_q.eval()
print("Loaded pcd query model")

best_swin_model = "/home/matej/diploma/Diploma/experiments/PMF-SemanticKitti/log_SemanticKitti_PMFNet-swin_tiny_bs2-lr0.0001_swin_no_dropout_batch_8_img_q/checkpoint/best_IOU_model.pth"
swin_model_img_q =  pc_processor.models.FusionCrossNet(backbone="swin_tiny", pcd_q=False, size=7)
state_dict = torch.load(best_swin_model, map_location="cpu")
swin_model_img_q.load_state_dict(state_dict)
swin_model_img_q.cuda()
swin_model_img_q.eval()
print("Loaded img query model")

# print number of parameters for each model
# print("PMF model parameters: ", sum(p.numel() for p in pmf.parameters() if p.requires_grad))
# print("Swin model parameters: ", sum(p.numel() for p in swin_model.parameters() if p.requires_grad))

color_data = color_config["color_map"]
color_map = np.zeros((max(color_data.keys())+1, 3), dtype=np.uint8)
for key, value in color_data.items():
    color_map[key] = value

def colorize(image):
    image = valset.class_map_lut_inv[image].astype(np.uint8)
    image = color_map[image]
    return image

def remove_padding(image, padding=30):
    return image[5:-5, 30:-30]

import matplotlib.pyplot as plt
from PIL import Image

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
        
        #patches, _ = patches_model1(img_feature, pcd_feature)
        pmf_out, _ = pmf(pcd_feature, img_feature)
        swin_pcd_q, _ = swin_model_pcd_q(pcd_feature, img_feature)
        swin_img_q, img = swin_model_img_q(pcd_feature, img_feature)
        #theirs, _ = pmf(pcd_feature, img_feature)

        # plt.imshow(valset.sem_color_lut[input_label[0].cpu().detach().numpy()])
        # plt.show()
        # diff = (swin.argmax(1)[label_mask] != input_label[label_mask])
        # diff_i = torch.zeros_like(input_label).float()
        # diff_i[label_mask] = diff.float()

        im = (img_feature[0].permute(1,2,0).cpu().detach().numpy() * 255).astype(np.uint8)

        label = colorize(input_label[0].cpu().detach().numpy())
        img_out = colorize(img[0].argmax(0).cpu().detach().numpy())
        swin_pcd_q_col = colorize(swin_pcd_q[0].argmax(0).cpu().detach().numpy())
        swin_img_q_col = colorize(swin_img_q[0].argmax(0).cpu().detach().numpy())
        pmf_col = colorize(pmf_out[0].argmax(0).cpu().detach().numpy())

        im = remove_padding(im)
        img_out = remove_padding(img_out)
        label = remove_padding(label)
        swin_pcd_q_col = remove_padding(swin_pcd_q_col)
        swin_img_q_col = remove_padding(swin_img_q_col)
        pmf_col = remove_padding(pmf_col)

        #stack all images together with 10px white space between them
        # im = np.vstack((im, np.ones((10, im.shape[1], 3), dtype=np.uint8) * 255, label, np.ones((10, im.shape[1], 3), dtype=np.uint8) * 255, pmf_col, np.ones((10, im.shape[1], 3), dtype=np.uint8) * 255, swin_img_q_col,  np.ones((10, im.shape[1], 3), dtype=np.uint8) * 255, img_out))
        # im = Image.fromarray(im)
        # im.save(f"../slike/{i:02d}.png")

        im = Image.fromarray(im)
        im.save(f"../slike1/barvna{i}.png")

        im = Image.fromarray(label)
        im.save(f"../slike1/label{i}.png")

        im = Image.fromarray(swin_img_q_col)
        im.save(f"../slike1/scf{i}.png")

        im = Image.fromarray(pmf_col)
        im.save(f"../slike1/pmf{i}.png")

        plt.subplot(2,2,1)
        plt.imshow(img_out)
        plt.title("Img out")
        plt.axis("off")

        plt.subplot(2,2,2)
        plt.imshow(swin_img_q_col)
        plt.title("Image Query")
        plt.axis("off")

        plt.subplot(2,2,3)
        plt.imshow(pmf_col)
        plt.title("PMF")
        plt.axis("off")

        plt.subplot(2,2,4)
        plt.imshow(img_feature[0].permute(1,2,0).cpu().detach().numpy())
        #plt.imshow(colorize(input_label[0].cpu().detach().numpy()))
        plt.title("Image")
        plt.axis("off")
        plt.show()
        #exit()
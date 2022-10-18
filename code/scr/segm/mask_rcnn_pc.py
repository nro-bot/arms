import argparse
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import torch
from torchvision.models import detection
# from torchvision.utils import draw_segmentation_masks
from camera import Camera
import mask_rcnn_utils


def main(args):

    # setup mask r-cnn
    torch_device = torch.device("cpu")
    model = detection.maskrcnn_resnet50_fpn(pretrained=True).to(torch_device)
    model.eval()

    # setup camera
    c = Camera()
    c.start()
    c.setup_pointcloud()

    fig = plt.figure()

    try:
        while True:

            plt.clf()
            ax = fig.gca()

            pc, tex, color_image = c.get_pointcloud_and_texture()

            # Important! OpenCV uses BGR.
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            image_pt = torch.tensor(color_image / 255., dtype=torch.float32, device=torch_device)[None].permute((0, 3, 1, 2))

            with torch.no_grad():
                segm_d = model(image_pt)[0]

            for key, value in segm_d.items():
                segm_d[key] = value.cpu()

            if len(segm_d["labels"]) == 0:
                continue

            # I would need to filter by instances here. Doesn't display class labels.
            # segm = segm_d["masks"][:, 0]
            # images = draw_segmentation_masks((image_pt[0].cpu() * 255).type(torch.uint8), segm.cpu() > 0.5)
            # images = images.permute((1, 2, 0))
            # images = images.cpu().numpy()

            boxes = segm_d["boxes"].numpy()
            masks = segm_d["masks"][:, 0].numpy()
            labels = segm_d["labels"].numpy()
            scores = segm_d["scores"].numpy()

            m = scores > args.object_threshold

            boxes = boxes[m]
            masks = masks[m]
            labels = labels[m]
            scores = scores[m]

            masks = masks.transpose((1, 2, 0))
            masks = (masks > args.mask_threshold).astype(np.int32)

            n_objects = len(scores)
            total_mask = np.zeros_like(masks[:, :, 0], dtype=np.bool)
            colors = mask_rcnn_utils.random_colors(n_objects)

            for i in range(n_objects):

                color_image = mask_rcnn_utils.apply_mask(color_image, masks[:, :, i], colors[i], alpha=1)
                total_mask += masks[:, :, i].astype(np.bool)

            print(tex)

            plt.subplot(1, 2, 1)
            plt.imshow(color_image)
            plt.subplot(1, 2, 2)
            plt.imshow(total_mask)
            plt.show()

            total_mask = total_mask.transpose().reshape(-1)

            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            pcd = c.get_pointcloud_open3d(cam_vertices=pc[total_mask], cam_texture=tex[total_mask], image=color_image)
            o3d.visualization.draw_geometries_with_editing([pcd])

    finally:

        # Stop streaming
        c.stop()


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cpu")
parser.add_argument("-o", "--object-threshold", type=float, default=0.5)
parser.add_argument("-m", "--mask-threshold", type=float, default=0.2)
main(parser.parse_args())

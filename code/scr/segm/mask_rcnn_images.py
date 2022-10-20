import argparse
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision.models import detection
from torchvision.utils import draw_segmentation_masks
from camera import Camera
from mask_rcnn_utils import display_instances, coco_class_names


def main(args):

    # setup mask r-cnn

    torch_device = torch.device("cuda:0")
    model = detection.maskrcnn_resnet50_fpn(pretrained=True).to(torch_device)
    model.eval()

    # setup camera
    c = Camera()
    c.start()

    fig = plt.figure()

    try:
        while True:

            plt.clf()
            ax = fig.gca()

            frames = c.get_frame()

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

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

            display_instances(color_image, boxes, masks, labels, coco_class_names, scores=scores, show_bbox=False, ax=ax)
            plt.pause(0.1)

    finally:

        # Stop streaming
        c.stop()


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cpu")
parser.add_argument("-o", "--object-threshold", type=float, default=0.5)
parser.add_argument("-m", "--mask-threshold", type=float, default=0.2)
main(parser.parse_args())

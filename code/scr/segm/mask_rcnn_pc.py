import argparse
from optparse import Option
import time
from pyparsing import Opt
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
from threading import Lock, Thread, Event
import queue
import cv2
import open3d as o3d
import torch
from torchvision.models import detection
# from torchvision.utils import draw_segmentation_masks
from camera import Camera
import mask_rcnn_utils
import utils


def perception_on_point_cloud(model, torch_device, object_threshold, mask_threshold, pc, tex, color_image):

    # Important! OpenCV uses BGR.
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    image_pt = torch.tensor(color_image / 255., dtype=torch.float32, device=torch_device)[None].permute((0, 3, 1, 2))

    with torch.no_grad():
        segm_d = model(image_pt)[0]

    for key, value in segm_d.items():
        segm_d[key] = value.cpu()

    if len(segm_d["labels"]) == 0:
        return None, None, False

    boxes = segm_d["boxes"].numpy()
    masks = segm_d["masks"][:, 0].numpy()
    labels = segm_d["labels"].numpy()
    scores = segm_d["scores"].numpy()

    m = scores > object_threshold

    if np.sum(m) == 0:
        return None, None, False

    boxes = boxes[m]
    masks = masks[m]
    labels = labels[m]
    scores = scores[m]

    masks = masks.transpose((1, 2, 0))
    masks = (masks > mask_threshold).astype(np.int32)

    n_objects = len(scores)
    total_mask = np.zeros_like(masks[:, :, 0], dtype=np.bool)
    colors = mask_rcnn_utils.random_colors(n_objects)

    for i in range(n_objects):

        color_image = mask_rcnn_utils.apply_mask(color_image, masks[:, :, i], colors[i], alpha=0.2)
        total_mask += masks[:, :, i].astype(np.bool)

    u, v, mask1 = utils.tex_coords_to_u_v_mask(tex, color_image)

    colors = color_image[u, v] / 255.
    mask2 = total_mask[u, v]

    pc = pc[mask1]
    colors = colors[mask1]
    mask2 = mask2[mask1]

    pc = pc[mask2]
    colors = colors[mask2]

    return pc, colors, True


def run_perception(input_queue, output_queue, args):

    while True:

        item = input_queue.get()

        if item is None:
            break

        out = perception_on_point_cloud(*args, *item)

        output_queue.put(out)


def main(args):

    # setup mask r-cnn
    torch_device = torch.device("cuda:0")
    model = detection.maskrcnn_resnet50_fpn(pretrained=True).to(torch_device)
    model.eval()

    # setup camera
    camera = Camera()
    camera.start()
    camera.setup_pointcloud()

    # setup point cloud
    pcd = o3d.geometry.PointCloud()
    pcd_added = False

    # setup vis
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # setup a thread for Mask R-CNN
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    t = Thread(target=run_perception, args=(input_queue, output_queue, (model, torch_device, args.object_threshold, args.mask_threshold)))
    t.start()

    # update vis
    close = False
    while not close:

        # get a frame and send it for processing
        frame = camera.get_pointcloud_and_texture()
        input_queue.put(frame)
        
        # update the visualization while we wait
        while not close:

            if not output_queue.empty():
                # update point cloud, get new frame
                item = output_queue.get()
                pc, colors, flag = item
                if not flag:
                    # processing failed (probably didn't find objects in the image)
                    break
                utils.update_open3d_pointcloud(pcd, pc, colors)
                if not pcd_added:
                    # first time showing a point cloud
                    vis.add_geometry(pcd)
                    pcd_added = True
                else:
                    vis.update_geometry(pcd)
                break

            vis.update_renderer()
            close = close or (not vis.poll_events())

            time.sleep(1 / 30)

    # clean-up
    # wait for Mask R-CNN to process the last frame
    while not input_queue.empty():
        time.sleep(0.1)
    # get the result
    while not output_queue.empty():
        output_queue.get()
    # send a stop signal
    input_queue.put(None)
    t.join()
    camera.stop()
    vis.destroy_window()


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cpu")
parser.add_argument("-o", "--object-threshold", type=float, default=0.5)
parser.add_argument("-m", "--mask-threshold", type=float, default=0.2)
main(parser.parse_args())

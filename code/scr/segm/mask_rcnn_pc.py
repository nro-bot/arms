import argparse
import time
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


def run_perception(camera, model, torch_device, object_threshold, mask_threshold, pcd, dirty, lock):

    pc, tex, color_image = camera.get_pointcloud_and_texture()

    # Important! OpenCV uses BGR.
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    image_pt = torch.tensor(color_image / 255., dtype=torch.float32, device=torch_device)[None].permute((0, 3, 1, 2))

    with torch.no_grad():
        segm_d = model(image_pt)[0]

    for key, value in segm_d.items():
        segm_d[key] = value.cpu()

    if len(segm_d["labels"]) == 0:
        return

    # I would need to filter by instances here. Doesn't display class labels.
    # segm = segm_d["masks"][:, 0]
    # images = draw_segmentation_masks((image_pt[0].cpu() * 255).type(torch.uint8), segm.cpu() > 0.5)
    # images = images.permute((1, 2, 0))
    # images = images.cpu().numpy()

    boxes = segm_d["boxes"].numpy()
    masks = segm_d["masks"][:, 0].numpy()
    labels = segm_d["labels"].numpy()
    scores = segm_d["scores"].numpy()

    m = scores > object_threshold

    if np.sum(m) == 0:
        return

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

    cam_texture = np.stack([tex[:, 1], tex[:, 0]], axis=-1)
    cam_texture = cam_texture * (color_image.shape[0], color_image.shape[1]) + 0.5
    cam_texture = cam_texture.astype(np.uint32)
    u, v = cam_texture[:, 0], cam_texture[:, 1]

    mask = np.logical_and(
        np.logical_and(0 <= u, u <= color_image.shape[0] - 1),
        np.logical_and(0 <= v, v <= color_image.shape[1] - 1)
    )
    mask = np.logical_not(mask)
    u = np.clip(u, 0, color_image.shape[0] - 1)
    v = np.clip(v, 0, color_image.shape[1] - 1)

    # open3d expects 0 - 1 image, realsense returns 0 - 255
    colors = color_image[u, v] / 255.
    colors[mask] = 0.

    segm_mask = total_mask[u, v]
    # plt.subplot(1, 2, 1)
    # plt.imshow(color_image)
    # plt.subplot(1, 2, 2)
    # plt.imshow(total_mask)
    # plt.show()

    print("y")
    lock.acquire(blocking=True)
    pcd.points = o3d.utility.Vector3dVector(pc[segm_mask])
    pcd.colors = o3d.utility.Vector3dVector(colors[segm_mask])
    dirty.set()
    lock.release()
    print("z")


def main(args):

    # setup mask r-cnn
    torch_device = torch.device("cuda:0")
    model = detection.maskrcnn_resnet50_fpn(pretrained=True).to(torch_device)
    model.eval()

    # setup camera
    c = Camera()
    c.start()
    c.setup_pointcloud()

    # setup point cloud
    pcd = o3d.geometry.PointCloud()

    # setup vis
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.update_renderer()

    lock = Lock()
    dirty = Event()

    i = 0

    # start worker thread
    # t = Thread(target=run_perception, args=(c, model, torch_device, args.object_threshold, args.mask_threshold, vis, pcd, dirty, lock))
    # t.start()

    # update vis
    while True:

        run_perception(c, model, torch_device, args.object_threshold, args.mask_threshold, pcd, dirty, lock)
        print("x")

        lock.acquire(blocking=True)

        if dirty.is_set():
            print("zz")
            if i == 0:
                vis.remove_geometry(pcd)
                vis.add_geometry(pcd)
            else:
                vis.update_geometry(pcd)
            dirty.clear()
            i += 1

        vis.update_renderer()
        close = vis.poll_events()
        lock.release()

        if not close:
            break

        time.sleep(1 / 30)

    # clean-up
    c.stop()
    vis.destroy_window()


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cpu")
parser.add_argument("-o", "--object-threshold", type=float, default=0.5)
parser.add_argument("-m", "--mask-threshold", type=float, default=0.2)
main(parser.parse_args())

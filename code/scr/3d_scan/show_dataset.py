import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import utils


def main(args):

    with open(args.save_file, "rb") as f:
        data = pickle.load(f)

    for i in range(len(data)):
        pcd = data[i]["clouds"]
        # pcd = pcd[np.random.randint(len(pcd), size=4000)]

        plt.subplot(1, 2, 1)
        plt.imshow(data[i]["images"])
        plt.subplot(1, 2, 2)
        plt.imshow(data[i]["depths"] / np.max(data[i]["depths"]))
        plt.show()

        pcd = utils.create_open3d_pointcloud(pcd.astype(np.float32))
        o3d.visualization.draw_geometries_with_editing([pcd])

        # Showing ordered point cloud.
        # pcd = data[i]["clouds"][0]
        # pcd = pcd.reshape(720, 1280, 3)
        # plt.subplot(1, 2, 1)
        # plt.imshow(data[i]["images"][0])
        # plt.subplot(1, 2, 2)
        # plt.imshow(pcd[:, :, 0])
        # plt.show()


parser = argparse.ArgumentParser("Find objects for a particular task.")
parser.add_argument("save_file")
main(parser.parse_args())

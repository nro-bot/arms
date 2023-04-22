import argparse
import pickle
import pyrealsense2 as rs
import numpy as np
import cv2
from camera import Camera
import paths, utils


def main(args):

    c = Camera()
    c.start()
    c.setup_pointcloud()

    if args.transform:
        c.load_calibration(paths.DEFAULT_CALIBRATION_PATH)

    try:
        data = []
        while True:

            input("Take picture? ")

            # Wait for a coherent pair of frames: depth and color
            frames = c.get_frame()
            depth_image, color_image = utils.realse_frame_to_numpy(frames)
            vertx, texcoords, _ = c.get_pointcloud_and_texture(frame=frames)

            if args.transform:
                vertx = utils.coord_transform(c.cam2world, vertx)
                print(c.cam2world)

            d = {
                "depths": depth_image,
                "images": color_image,
                "clouds": vertx,
                "texcoords": texcoords,
            }
            data.append(d)

            with open(args.save_file, "wb") as f:
                pickle.dump(data, f)

    finally:

        # Stop streaming
        c.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Find objects for a particular task.")
    parser.add_argument("save_file")
    parser.add_argument("-t", "--transform", default=False, action="store_true")
    main(parser.parse_args())

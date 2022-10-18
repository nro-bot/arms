import argparse
import pickle
import open3d as o3d
from camera import Camera
import utils


def view(pcd):

    o3d.visualization.draw_geometries([pcd])
    x = input("Save? (yes/no): ")
    if x.strip().lower() == "yes":
        return True

    return False


def main(args):

    c = Camera()
    c.start()
    c.setup_pointcloud()

    if args.raw:
        # save numpy arrays that are used to create the pointcloud
        vertices, colors = c.get_pointcloud_open3d(return_raw=True)
        if args.view:
            # view pointcloud before saving it
            while True:
                x = view(utils.create_open3d_pointcloud(vertices, colors))
                if x:
                    break
                vertices, colors = c.get_pointcloud_open3d(return_raw=True)

        with open(args.save_path, "wb") as f:
            pickle.dump({
                "vertices": vertices,
                "colors": colors
            }, f)
    else:
        # save pointcloud in a standard format (e.g. .pcd) specified by the save path
        pcd = c.get_pointcloud_open3d()
        if args.view:
            # view pointcloud before saving it
            while True:
                x = view(pcd)
                if x:
                    break
                pcd = c.get_pointcloud_open3d()

        o3d.io.write_point_cloud(args.save_path, pcd)


parser = argparse.ArgumentParser()
parser.add_argument(
    "save_path",
    help="Where to store the pointcloud. Manually specify that this is .pcd file. "
         "You can also use other file types supported by o3d."
)
parser.add_argument(
    "-r", "--raw", default=False, action="store_true",
    help="This will store a pickle consisting of two numpy arrays instead."
)
parser.add_argument(
    "-v", "--view", default=False, action="store_true",
    help="View the point cloud before saving it."
)
main(parser.parse_args())

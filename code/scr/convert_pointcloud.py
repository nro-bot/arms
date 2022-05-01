import argparse
import open3d as o3d


def main(args):

    pcd = o3d.io.read_point_cloud(args.load_path)
    o3d.io.write_point_cloud(args.save_path, pcd)


parser = argparse.ArgumentParser("Specify a save path with different extension to convert point cloud format.")
parser.add_argument("load_path")
parser.add_argument("save_path")
main(parser.parse_args())

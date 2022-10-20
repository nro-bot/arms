import argparse
import open3d as o3d


def main(args):

    pcd = o3d.io.read_point_cloud(args.load_path)
    o3d.visualization.draw_geometries_with_editing([pcd])


parser = argparse.ArgumentParser(
    "Interactive visualizer. Press 'k' to begin selection. "
    "Draw a rectangle or press 'ctrl + left click' to draw a polygon. "
    "Press 'c' to crop and save the new point cloud."
)
parser.add_argument("load_path")
main(parser.parse_args())

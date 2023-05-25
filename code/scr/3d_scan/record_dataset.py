import argparse
import pickle
import pyrealsense2 as rs
import numpy as np

from camera import Camera
import paths, utils


def main(args):

    data = []

    c = Camera()
    c.load_calibration(paths.DEFAULT_CALIBRATION_PATH)

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    color_profile = profile.get_stream(rs.stream.color)
    depth_profile = profile.get_stream(rs.stream.depth)

    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    color_distortion_coeffs = np.array(color_intrinsics.coeffs, dtype=np.float32)

    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
    depth_distortion_coeffs = np.array(depth_intrinsics.coeffs, dtype=np.float32)

    color_fx = color_intrinsics.fx
    color_fy = color_intrinsics.fy
    color_ppx = color_intrinsics.ppx
    color_ppy = color_intrinsics.ppy

    depth_fx = depth_intrinsics.fx
    depth_fy = depth_intrinsics.fy
    depth_ppx = depth_intrinsics.ppx
    depth_ppy = depth_intrinsics.ppy

    cam_d = {
        "color_fx": color_fx,
        "color_fy": color_fy,
        "color_ppx": color_ppx,
        "color_ppy": color_ppy,
        "color_distortion_coeffs": color_distortion_coeffs,
        "depth_fx": depth_fx,
        "depth_fy": depth_fy,
        "depth_ppx": depth_ppx,
        "depth_ppy": depth_ppy,
        "depth_distortion_coeffs": depth_distortion_coeffs,
        "cam2world": c.cam2world,
    }

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to other frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:

            input("Take picture? ")

            # Get the aligned frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                raise ValueError("Frames are not valid!")

            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Create point cloud. We want to project our depth image into 3D space
            pc = rs.pointcloud()
            pc.map_to(color_frame)

            # Generate the pointcloud and texture mappings
            points = pc.calculate(aligned_depth_frame)
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

            if args.transform:
                verts = utils.coord_transform(c.cam2world, verts)
                print(c.cam2world)

            d = {
                "depths": depth_image,
                "images": color_image,
                "clouds": verts,
                "texcoords": texcoords,
                **cam_d
            }
            data.append(d)

            with open(args.save_file, "wb") as f:
                pickle.dump(data, f)
    finally:
        # Don't forget to stop the pipeline at the end of the script
        pipeline.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Find objects for a particular task.")
    parser.add_argument("save_file")
    parser.add_argument("-t", "--transform", default=False, action="store_true")
    main(parser.parse_args())

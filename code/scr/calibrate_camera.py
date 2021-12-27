import argparse
from camera import Camera
import paths


def main(args):

    c = Camera()
    c.start()

    try:
        c.calibrate(show=args.show)
        c.save_calibration(args.save_path)

    finally:

        # Stop streaming
        c.stop()


parser = argparse.ArgumentParser()
parser.add_argument("--save-path", default=paths.DEFAULT_CALIBRATION_PATH,
                    help="Save path for pickle with camera calibration data.")
parser.add_argument("--show", default=False, action="store_true", help="Show intermediate steps of calibration.")
main(parser.parse_args())

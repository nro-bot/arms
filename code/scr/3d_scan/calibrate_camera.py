import argparse
import time
from camera import Camera
import paths
import exceptions


def main(args):

    print("Make sure the checkerboard is aligned with the robot.")
    c = Camera()
    c.start()
    time.sleep(1)

    save_path = paths.SCAN_3D_CALIBRATION_TEMPLATE.format(args.camera)

    try:
        for i in range(5):
            try:
                c.calibrate(show=args.show)
                c.save_calibration(save_path)
                print("Checkerboard found, calibration data saved.")
                break
            except exceptions.CheckerboardNotFound:
                print("Checkerboard not found.")
                time.sleep(0.1)
                continue

    finally:

        # Stop streaming
        c.stop()


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--camera", default=0, type=int)
parser.add_argument("--show", default=False, action="store_true", help="Show intermediate steps of calibration.")
main(parser.parse_args())

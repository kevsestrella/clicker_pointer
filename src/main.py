import os
import time
import logging as log
from argparse import ArgumentParser

import cv2

from input_feeder import InputFeeder
from face_detection import FaceDetector

def build_argparser():
    parser =  ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
            help="Path to Face Detection model xml")
    parser.add_argument("-i", "--input", required=True, type=str,
            help="Path to image/video file or CAM")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                    help="Specify the target device to infer on: "
                         "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                         "will look for a suitable plugin for device "
                         "specified (CPU by default)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                    help="Probability threshold for detections filtering"
                         "(0.5 by default)")
    parser.add_argument("-e", "--extensions", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    return parser

def main():
    args = build_argparser().parse_args()
    input_file = args.input
    logger = log.getLogger()
    if input_file == "CAM":
        input_feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(input_file):
            logger.error("Path should be file")
            exit(1)
        input_feeder = InputFeeder("video", input_file)

    face_detector = FaceDetector(args.face_detection_model, device=args.device, threshold=args.threshold,
            extensions=args.extensions)
    start_time = time.time()
    face_detector.load_model()
    logger.info("Face Detection Model loading time:{:.3f} ms".format((time.time() - start_time) * 1000))
    input_feeder.load_data()

    counter = 0
    for flag, frame in input_feeder.next_batch():
        if not flag:
            break
        pressed_key = cv2.waitKey(60)
        counter = counter + 1
        face_coordinates, face_image = face_detector.predict(frame.copy())
        if face_coordinates == 0:
            continue
        if pressed_key == 27:
            logger_object.error("exit key is pressed..")
            break

if __name__ == "__main__":
    main()

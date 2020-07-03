import os
import time

# import logging as log
# log.basicConfig(level=log.INFO)

from argparse import ArgumentParser

import cv2

from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetector
from facial_landmarks_detection import FaceLandmarkDetector
from head_pose_estimation import HeadPoseEstimator
from gaze_estimation import GazeEstimator

def build_argparser():
    parser =  ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
            help="Path to Face Detection model xml")
    parser.add_argument("-fl", "--face_landmark_model", required=True, type=str,
            help="Path to Face Landmark Detection model xml")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
            help="Path to Head Pose Estimation model xml")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
            help="Path to Head Pose Estimation model xml")
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
    if input_file == "CAM":
        input_feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(input_file):
            logger.error("Path should be file")
            exit(1)
        input_feeder = InputFeeder("video", input_file)

    face_detector = FaceDetector(args.face_detection_model, device=args.device, threshold=args.threshold,
            extensions=args.extensions)
    face_landmark_detector = FaceLandmarkDetector(args.face_landmark_model, device=args.device, threshold=args.threshold,
            extensions=args.extensions)
    head_pose_estimator = HeadPoseEstimator(args.head_pose_model, device=args.device, threshold=args.threshold,
            extensions=args.extensions)
    gaze_estimator = GazeEstimator(args.gaze_estimation_model, device=args.device, threshold=args.threshold,
            extensions=args.extensions)
    mouse_controller = MouseController('medium', 'fast')

    face_detector.load_model()
    face_landmark_detector.load_model()
    head_pose_estimator.load_model()
    gaze_estimator.load_model()

    input_feeder.load_data()

    width = 500
    height = int(width*9/16)

    for flag, frame in input_feeder.next_batch():

        if not flag:
            break
        pressed_key = cv2.waitKey(60)

        face_coordinates, face_image = face_detector.predict(frame.copy())
        left_eye_img, righ_eye_img, eye_coords = face_landmark_detector.predict(face_image)
        head_pose_estimate = head_pose_estimator.predict(face_image)
        mouse_coordinate, gaze_vector = gaze_estimator.predict(left_eye_img, righ_eye_img, head_pose_estimate)

        cv2.imshow('frame', cv2.resize(frame, (width, height)))
        mouse_controller.move(mouse_coordinate[0], mouse_coordinate[1])

        if pressed_key == 27:
            logger_object.error("exit key is pressed..")
            break

if __name__ == "__main__":
    main()

import os
import math
import logging as log
log.basicConfig(level=log.INFO)
from argparse import ArgumentParser

import cv2
import numpy as np

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
                         "CPU, GPU, FPGA or MYRIAD is acceptable(CPU by default)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                    help="Probability threshold for detections filtering"
                         "(0.5 by default)")
    parser.add_argument("-e", "--extensions", required=False, type=str,
                        default=None,
                        help="Extensions for custom layers support")
    parser.add_argument("-flag", "--visualization", type=str, required=False, nargs='+',
                        default=[],
                        help="Example: --flag [fd] [fl] [hp] [ge] (No flag is required,each represents stages of the pipeline)")
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

    width = 1000
    height = int(width*9/16)

    for flag, frame in input_feeder.next_batch():

        if not flag:
            break
        pressed_key = cv2.waitKey(60)

        face_detected = face_detector.predict(frame)
        if face_detected:
            face_coordinates, face_image = face_detected
            if not face_coordinates:
                continue
        else:
            continue
        if "fd" in args.visualization:
            cv2.rectangle(frame, (face_coordinates[0], face_coordinates[1]), (face_coordinates[2], face_coordinates[3]), (36, 255, 12), 2)
            cv2.putText(frame, "Face Detected", (face_coordinates[0], face_coordinates[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        left_eye_img, righ_eye_img, eye_coords = face_landmark_detector.predict(face_image)
        if "fl" in args.visualization:
            frame_eye_coords_min = np.array(eye_coords)[:, :2] + np.array(face_coordinates)[:2]
            frame_eye_coords_max = np.array(eye_coords)[:, 2:] + np.array(face_coordinates)[:2]
            cv2.rectangle(frame, (frame_eye_coords_min[0][0], frame_eye_coords_min[0][1]), (frame_eye_coords_max[0][0], frame_eye_coords_max[0][1]), (36, 255,12), 2)
            cv2.rectangle(frame, (frame_eye_coords_min[1][0], frame_eye_coords_min[1][1]), (frame_eye_coords_max[1][0], frame_eye_coords_max[1][1]), (36, 255, 12), 2)

        head_pose_estimate = head_pose_estimator.predict(face_image)
        if "hp" in args.visualization:
            cv2.putText(frame,
                        "yaw:{:.1f}|pitch:{:.1f}|roll:{:.1f}".format(*head_pose_estimate),
                        (20, 35), cv2.FONT_HERSHEY_COMPLEX, 1.2, (36, 255, 12), 3)

        mouse_coordinate, gaze_vector = gaze_estimator.predict(left_eye_img, righ_eye_img, head_pose_estimate)
        if "ge" in args.visualization:
            head_pose_estimate = np.array(head_pose_estimate)
            yaw, pitch, roll = head_pose_estimate * np.pi / 180.0

            focal_length = 950
            scale = 100

            origin = (int(face_coordinates[0]+(face_coordinates[2]-face_coordinates[0])/2), int(face_coordinates[1]+(face_coordinates[3]-face_coordinates[1])/2))

            r_x = np.array([[1, 0, 0],
                            [0, math.cos(pitch), -math.sin(pitch)],
                            [0, math.sin(pitch), math.cos(pitch)]])
            r_y = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                            [0, 1, 0],
                            [math.sin(yaw), 0, math.cos(yaw)]])
            r_z = np.array([[math.cos(roll), -math.sin(roll), 0],
                            [math.sin(roll), math.cos(roll), 0],
                            [0, 0, 1]])
            r = r_z @ r_y @ r_x

            zaxis = np.array(([0, 0, -1 * scale]), dtype='float32')
            offset = np.array(([0, 0, focal_length]), dtype='float32')
            zaxis = np.dot(r, zaxis) + offset
            tip = (int(zaxis[0] / zaxis[2] * focal_length) + origin[0], int(zaxis[1] / zaxis[2] * focal_length) + origin[1])

            cv2.arrowedLine(frame, origin, tip, (0, 0, 255), 3, tipLength=0.3)

        cv2.imshow('frame', cv2.resize(frame, (width, height)))
        mouse_controller.move(mouse_coordinate[0], mouse_coordinate[1])

        if pressed_key == 27:
            logger.error("exit key is pressed..")
            break


if __name__ == "__main__":
    main()

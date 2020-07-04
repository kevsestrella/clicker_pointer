import math

import cv2

from model import Model_X
from timer import timeit


class GazeEstimator(Model_X):
    """
    Class for the Face Detection Model.
    """

    def __init__(self, model_name, device="CPU", threshold=0.5, extensions=None):
        super().__init__(
            model_name, device=device, threshold=threshold, extensions=extensions
        )

    @timeit
    def predict(self, left_eye_image, right_eye_image, head_pose_estimate):
        """
        This method is meant for running predictions on the input image.
        """
        self.input_shape = (60, 60, 60, 60)
        outputs = self.exec_network.infer(
            inputs={
                "left_eye_image": self.preprocess_input(left_eye_image),
                "right_eye_image": self.preprocess_input(right_eye_image),
                "head_pose_angles": head_pose_estimate,
            }
        )

        return self.preprocess_output(outputs, head_pose_estimate)

    def preprocess_output(self, outputs, head_pose_estimate):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """

        roll = head_pose_estimate[2]
        outputs = outputs[self.output_name][0]
        cos_theta = math.cos(roll * math.pi / 180)
        sin_theta = math.sin(roll * math.pi / 180)
        x = outputs[0] * cos_theta + outputs[1] * sin_theta
        y = outputs[1] * cos_theta - outputs[0] * sin_theta

        return (x, y), outputs

from model import Model_X


class FaceLandmarkDetector(Model_X):
    """
    Class for the Face Detection Model.
    """

    def __init__(self, model_name, device="CPU", threshold=0.5, extensions=None):
        super().__init__(
            model_name, device=device, threshold=threshold, extensions=extensions
        )

    def preprocess_output(self, outputs, image):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        outputs = outputs[self.output_name][0]

        left_eye_x = int(outputs[0] * image.shape[1])
        left_eye_y = int(outputs[1] * image.shape[0])
        right_eye_x = int(outputs[2] * image.shape[1])
        right_eye_y = int(outputs[3] * image.shape[0])

        left_eye = (left_eye_x - 10, left_eye_y - 10, left_eye_x + 10, left_eye_y + 10)
        right_eye = (
            right_eye_x - 10,
            right_eye_y - 10,
            right_eye_x + 10,
            right_eye_y + 10,
        )

        return (
            image[left_eye[0] : left_eye[2], left_eye[1] : left_eye[3]],
            image[right_eye[0] : right_eye[2], right_eye[1] : right_eye[3]],
            (left_eye, right_eye),
        )

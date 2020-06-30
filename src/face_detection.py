from model import Model_X
import logging as log

import cv2

from openvino.inference_engine import IENetwork, IECore


class FaceDetector(Model_X):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold=0.5, extensions=None):
        super().__init__(model_name, device=device, threshold=threshold, extensions=extensions)


    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        outputs = self.exec_network.infer({self.input_name: self.preprocess_input(image)})
        coordinates = self.preprocess_output(outputs, image.shape)

        face_image = image[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
        return coordinates, face_image


    def preprocess_output(self, outputs, image_shape):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        for box in outputs[self.output_name][0][0]:
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * image_shape[1])
                ymin = int(box[4] * image_shape[0])
                xmax = int(box[5] * image_shape[1])
                ymax = int(box[6] * image_shape[0])
                # limited to only one face detection
                return [xmin, ymin, xmax, ymax]

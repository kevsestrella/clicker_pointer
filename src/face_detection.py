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


    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        for box in outputs[self.output_name][0][0]:
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                # limited to only one face detection
                return [xmin, ymin, xmax, ymax], image[ymin:ymax, xmin:xmax]

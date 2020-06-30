import logging as log

import cv2

from openvino.inference_engine import IENetwork, IECore


class FaceDetector:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold=0.5, extensions=None):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold
        self.extensions = extensions
        self.face_image = None
        self.face_coordinates = None
        self.exec_network = None


    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.model = IENetwork(self.model_structure, self.model_weights)
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

        self.plugin = IECore()

        if self.extensions and "CPU" in self.device:
            self.plugin.add_extension(self.extensions, self.device)

        supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]

        if unsupported_layers:
            log.error(f"Unsupported layers found: {unsupported_layers}")
            exit(1)

        self.exec_network = self.plugin.load_network(self.model, self.device)

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        outputs = self.exec_network.infer({self.input_name: self.preprocess_input(image)})
        coordinates = self.preprocess_output(outputs, image.shape)

        face_image = image[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
        return coordinates, face_image


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        ret = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        ret = ret.transpose((2, 0, 1))
        ret = ret.reshape(1, *ret.shape)

        return ret

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

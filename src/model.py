from abc import ABC, abstractmethod
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore
from timer import timeit


class Model_X(ABC):
    '''
    Abstract Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold=0.5, extensions=None):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold
        self.extensions = extensions

    @timeit
    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.network = IENetwork(self.model_structure, self.model_weights)
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_name = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape

        self.core = IECore()

        if self.extensions and "CPU" in self.device:
            self.core.add_extension(self.extensions, self.device)

        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        if unsupported_layers:
            log.error(f"Unsupported layers found: {unsupported_layers}")
            exit(1)

        self.exec_network = self.core.load_network(self.network, self.device)

    @timeit
    def predict(self, image):
        '''
        This abstractmethod is meant for running predictions on the input image.
        '''
        outputs = self.exec_network.infer({self.input_name: self.preprocess_input(image)})
        return self.preprocess_output(outputs, image)

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

    @abstractmethod
    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        pass

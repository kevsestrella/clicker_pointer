import cv2

from model import Model_X


class HeadPoseEstimator(Model_X):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold=0.5, extensions=None):
        super().__init__(model_name, device=device, threshold=threshold, extensions=extensions)


    def preprocess_output(self, outputs, image=None):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        output = []
        output.append(outputs['angle_y_fc'].tolist()[0][0])
        output.append(outputs['angle_p_fc'].tolist()[0][0])
        output.append(outputs['angle_r_fc'].tolist()[0][0])

        return output

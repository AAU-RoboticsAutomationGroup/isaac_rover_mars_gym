import cv2
from isaacgym import gymapi

class visualize:
    '''
    Class for visualising data from ExoMy robots.
    '''
    def __init__(self):
        cv2.namedWindow("Sensor")

    def show_image(self, env, cam_handle, gym, sim, filename, imagetype):
        '''
        Show image from camera. Image will be saved to file "filename"(Remember to add suffix).
        Imagetype is either *gymapi.IMAGE_COLOR* or *gymapi.IMAGE_DEPTH*.
        Cam_handle and env are single handles! Get environment nr. by [X].
        Remember to render cameras before visualisation.
        '''
        gym.write_camera_image_to_file(sim, env, cam_handle, imagetype, filename)
        im = cv2.imread(filename)
        cv2.imshow('Sensor',im)
        cv2.waitKey(1)

    def show_pointcloud():
        pass
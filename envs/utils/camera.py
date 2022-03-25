from isaacgym import gymutil, gymtorch, gymapi
import numpy as np

class camera:
    '''
    Camera class for adding cameras to the ExoMy robots. 
    Instanciate in the *_create_envs* method
    '''

    def __init__(self):
        # Camera properties
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 424     #Pixels width
        self.camera_props.height = 240    #Pixels height
        self.camera_props.near_plane = 0.16
        self.camera_props.far_plane = 3
        #Placement of camera
        self.local_transform = gymapi.Transform()
        self.local_transform.p = gymapi.Vec3(0,0,0.01) #Units in meters, (X, Y, Z) - X-up/down, Y-Side, Z-forwards/backwards
        self.local_transform.r = gymapi.Quat.from_euler_zyx(np.radians(180.0), np.radians(-119.0), np.radians(0.0))
        self.camera_handles = []
        print("Camera class initialized!")


    
    def add_camera(self, env, gym, exo_handle):
        '''
        Add camera to an ExoMy instance using an ExoMy_handle
        '''
        # Create camera sensor
        camera_handle = gym.create_camera_sensor(env, self.camera_props)
        # Add handle to camera_handles
        self.camera_handles.append(camera_handle)
        # Get body handle from robot
        body_handle = gym.get_actor_rigid_body_handle(env, exo_handle, 18)
        # Attatch camera to body using handles
        gym.attach_camera_to_body(camera_handle, env, body_handle, self.local_transform, gymapi.FOLLOW_TRANSFORM)

    def create_static_sphere(self, gym, sim, env, exo_handle, collision_it):
        '''
        Create a static sphere at the location of the camera.
        Pause the simulation at first timestep to confirm location of sphere.
        '''
        # Set marker options and create sphere
        marker_options = gymapi.AssetOptions()
        marker_options.fix_base_link = True
        marker_asset = gym.create_sphere(sim, 0.02, marker_options)
        # Get body handle from robot
        body_handle = gym.get_actor_rigid_body_handle(env, exo_handle, 18)
        body_transform = gym.get_rigid_transform(env,body_handle)

        marker_handle = gym.create_actor(env, marker_asset, body_transform, "marker", collision_it, 1, 1)
        gym.set_rigid_body_color(env, marker_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))



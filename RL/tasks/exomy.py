
import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET

from utils.torch_jit_utils import *
from tasks.base.vec_task import VecTask

from isaacgym import gymutil, gymtorch, gymapi


class Exomy(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):

        self.cfg = cfg

        self.max_episode_length = 100
        self.cfg["env"]["numObservations"] = 3
        self.cfg["env"]["numActions"] = 12
        self.max_effort_vel = math.pi
        self.max_effort_pos = math.pi/2
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)




        cam_pos = gymapi.Vec3(-1.0, -0.6, 0.8)
        cam_target = gymapi.Vec3(1.0, 1.0, 0.15)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        
        
        #    - set up gravity
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -3.721  
        #    - call super().create_sim with device args (see docstring)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        #    - set time step length
        self.dt = self.sim_params.dt
        #    - setup asset
        self._create_exomy_asset()
        #    - create ground plane
        self._create_ground_plane()
        #    - set up environments
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        





    def _create_exomy_asset(self):
        pass


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the nroaml force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0,0.0,1.0)
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self,num_envs,spacing, num_per_row):
       # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = "../assets"
        exomy_asset_file = "urdf/exomy_model/urdf/exomy_model.urdf"
        
        # if "asset" in self.cfg["env"]:
        #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        #     asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        # asset_path = os.path.join(asset_root, asset_file)
        # asset_root = os.path.dirname(asset_path)
        # asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.armature = 0.01
        # use default convex decomposition params
        asset_options.vhacd_enabled = False

        print("Loading asset '%s' from '%s'" % (exomy_asset_file, asset_root))
        exomy_asset = self.gym.load_asset(self.sim, asset_root, exomy_asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(exomy_asset)
        print(self.num_dof)
        #################################################
        # get joint limits and ranges for Franka
        exomy_dof_props = self.gym.get_asset_dof_properties(exomy_asset)
        exomy_lower_limits = exomy_dof_props["lower"]
        exomy_upper_limits = exomy_dof_props["upper"]
        exomy_ranges = exomy_upper_limits - exomy_lower_limits
        exomy_mids = 0.5 * (exomy_upper_limits + exomy_lower_limits)
        exomy_num_dofs = len(exomy_dof_props)

        #################################################
        # set default DOF states
        default_dof_state = np.zeros(exomy_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = exomy_mids


        
        exomy_dof_props["driveMode"] = [
            gymapi.DOF_MODE_VEL, #0  #LFB bogie
            gymapi.DOF_MODE_POS,  #1  #LF POS
            gymapi.DOF_MODE_VEL,  #2  #LF DRIVE
            gymapi.DOF_MODE_POS,  #3  #LM POS
            gymapi.DOF_MODE_VEL,  #4  #LM DRIVE
            gymapi.DOF_MODE_VEL, #5  #MRB bogie
            gymapi.DOF_MODE_POS,  #6  #LR POS
            gymapi.DOF_MODE_VEL,  #7  #LR DRIVE
            gymapi.DOF_MODE_POS,  #8  #RR POS
            gymapi.DOF_MODE_VEL,  #9  #RR DRIVE
            gymapi.DOF_MODE_VEL, #10 #RFB bogie
            gymapi.DOF_MODE_POS,  #11 #RF POS 
            gymapi.DOF_MODE_VEL,  #12 #RF DRIVE
            gymapi.DOF_MODE_POS,  #13 #RM POS
            gymapi.DOF_MODE_VEL,  #14 #RM DRIVE
            gymapi.DOF_MODE_VEL,  #15 #SHIT EYE 1
            gymapi.DOF_MODE_VEL   #16 #SHIT EYE 2
        ]

        

        exomy_dof_props["stiffness"].fill(800.0)
        exomy_dof_props["damping"].fill(0.01)
        exomy_dof_props["friction"].fill(0.0)
        pose = gymapi.Transform()
        pose.p.z = 0.2
        # asset is rotated z-up by default, no additional rotations needed
        pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        self.exomy_handles = []
        self.envs = []
        


        for i in range(num_envs):
            # Create environment
            env0 = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env0)

            
            exomy0_handle = self.gym.create_actor(
                env0,  # Environment Handle
                exomy_asset,  # Asset Handle
                pose,  # Transform of where the actor will be initially placed
                "exomy",  # Name of the actor
                i,  # Collision group that actor will be part of
                1,  # Bitwise filter for elements in the same collisionGroup to mask off collision
            )
            self.exomy_handles.append(exomy0_handle)

            
    
            # Configure DOF properties
            # Set initial DOF states
            # gym.set_actor_dof_states(env0, exomy0_handle, default_dof_state, gymapi.STATE_ALL)
            # Set DOF control properties
            self.gym.set_actor_dof_properties(env0, exomy0_handle, exomy_dof_props)
            #print(self.gym.get_actor_dof_properties((env0, exomy0_handle))

    def reset_idx(self, env_ids):
        pass
        #Used to reset a single environment
        
    def compute_observations(self):
        pass

    def compute_rewards(self):
        pass


    def pre_physics_step(self, actions):
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        _actions = actions.to(self.device)

        # actions_tensor[::self.num_dof] = actions.to(self.device).squeeze()
        # #print(np.shape(_actions))
        #print(np.shape(_actions))
        # print(actions.size())

        DRV_LF_joint_dof_handle = self.gym.find_actor_dof_handle(self.envs[0], self.exomy_handles[0], "DRV_LF_joint")
        #max = 100
        #actions_tensor = actions.to(self.device).squeeze() * 400
        #actions_tensor[1::17]=_actions[:,0] * max # POS ALL MOTORS
        #actions_tensor[2::17]=_actions[:,1] * max # LEFT BROKEN BOGIE
        #actions_tensor[3::17]=_actions[:,2] * max # FOUR POS MOTOR CONTROL - NOT FL AND ML
        #actions_tensor[4::17]=_actions[:,3] * max  # LEFT BOGIE BROKEN
        #actions_tensor[6::17]=_actions[:,4] * max # ALL RIGHT POS MOTORS
        #actions_tensor[7::17]=-1* max#_actions[:,5] * max # RIGHT BOGIE AND RR DRIVERIGHT BOGIE
        #actions_tensor[8::17]=_actions[:,6] * max # POS FR AND MR
        #actions_tensor[9::17]=_actions[:,7] * max # SAME AS PREV
        #actions_tensor[11::17]=_actions[:,8] * max # MR POS working correctly
       # actions_tensor[12::17]= -1* max#_actions[:,9] * max # WORKING MR DRIVE
       # actions_tensor[13::17]=-1* max#_actions[:,10] * max #NOTHING
        #actions_tensor[14::17]=-0.2* max#_actions[:,11] * max #RIGHT BOGIE
        #actions_tensor[0] = 100 #BOTH REAR DRIVE
        speed = 10
        actions_tensor[3] = 0
        actions_tensor[2] = speed 
        actions_tensor[4] = speed 
        actions_tensor[7] = speed 
        actions_tensor[9] = speed 
        actions_tensor[12] = speed
        
        actions_tensor[14] = speed
        
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor)) #)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor)) #)
        #forces = gymtorch.unwrap_tensor(actions_tensor)
        #self.gym.set_dof_actuation_force_tensor(self.sim, forces)
        pass
        
    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1

        self.compute_observations()
        self.compute_rewards()

@torch.jit.script
def compute_exomy_reward():
    reward = 0
    reset = 0
    return reward, reset        

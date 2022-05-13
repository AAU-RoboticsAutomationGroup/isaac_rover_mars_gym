
from cgitb import reset
import math
import os
import random
import xml.etree.ElementTree as ET
from operator import pos

import numpy as np
import torch
#from utils.kinematics import Rover
import torchgeometry as tgm
from isaacgym import gymapi, gymtorch, gymutil
from scipy.spatial.transform import Rotation as R
from utils.exo_depth_observation import (exo_depth_observation, height_lookup,
                                         visualize_points)
from utils.heigtmap_distribution import heightmap_distribution, heightmap_overlay
from utils.kinematics import Ackermann
from utils.tensor_quat_to_euler import tensor_quat_to_eul
from utils.terrain_generation import *
from utils.torch_jit_utils import *

from tasks.base.vec_task import VecTask


class Exomy_actual(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):

        self.cfg = cfg
        #self.Kinematics = Rover()
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self._num_camera_inputs = 1080
        self._num_observations = 4
        self.cfg["env"]["numCamera"] = self._num_camera_inputs
        self.cfg["env"]["numObservations"] = self._num_observations + self._num_camera_inputs#*5
        
        self.cfg["env"]["numActions"] = 2
        self.max_effort_vel = 5.2
        self.max_effort_pos = math.pi/2
        self.rock_reset_distance_exomy = self.cfg["env"]["exomyResetDistance"]
        self.rock_reset_distance_target = self.cfg["env"]["targetResetDistance"]
        # reward scales
        self.rew_scales = {}
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"] 
        self.rew_scales["pos"] = self.cfg["env"]["learn"]["pos_reward"] 
        self.rew_scales["collision"] = self.cfg["env"]["learn"]["collision_reward"] 
        self.rew_scales["heading"] = self.cfg["env"]["learn"]["heading_contraint_reward"] 
        self.rew_scales["torque_driving"] = self.cfg["env"]["learn"]["torque_reward_driving"] 
        self.rew_scales["torque_steering"] = self.cfg["env"]["learn"]["torque_reward_steering"] 
        self.rew_scales["uprightness"] = self.cfg["env"]["learn"]["uprightness_reward"]
        self.rew_scales["motion_contraint"] = self.cfg["env"]["learn"]["motion_contraint_reward"]
        self.rew_scales["acceleration"] = self.cfg["env"]["learn"]["acceleration_reward"]
        self.rew_scales["vel_near_goal"] = self.cfg["env"]["learn"]["vel_near_goal_reward"]
        self.rew_scales["motion_contraint"] = self.cfg["env"]["learn"]["motion_contraint_reward"] 
        self.rew_scales["goal_angle"] = self.cfg["env"]["learn"]["goal_angle_reward"] 
        
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # Retrieves buffer for Actor root states.
        # position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        # Buffer has shape (num_environments, num_actors * 13).
        dofs_per_env = 15

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.location_tensor_gym = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.dof_force_tensor_gym = self.gym.acquire_dof_force_tensor(self.sim)
        
        # Convert buffer to vector, one is created for the robot and for the marker.
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 2, 13)
        #vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)

        # Convert gym tensors to pytorch tensors.
        self.location_tensor = gymtorch.wrap_tensor(self.location_tensor_gym)[0::20]
        self.dof_force_tensor = gymtorch.wrap_tensor(self.dof_force_tensor_gym)

        # Position vector for robot
        self.root_states = vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[:, 0:3]
        # Rotation of robot
        self.root_quats = self.root_states[:, 3:7]
        # Linear Velocity of robot
        self.root_linvels = self.root_states[:, 7:10]
        # Angular Velocity of robot
        self.root_angvels = self.root_states[:, 10:13]
        
        self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_root_positions[:, 2] = 0

        # Previous actions and torques
        self.actions_nn = torch.zeros((self.num_envs, self.cfg["env"]["numActions"], 3), device=self.device)
        self.actions_steering_motors = torch.zeros((3, self.num_envs, 6), device=self.device)
        self.actions_driving_motors = torch.zeros((3, self.num_envs, 6), device=self.device)

        self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor)
        #self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        #self.steering_torques = torch.zeros((self.num_envs, 6, 3), device=self.device)
        #self.driving_torques = torch.zeros((self.num_envs, 6, 3), device=self.device)

        # Marker position
        self.marker_states = vec_root_tensor[:, 1, :]
        self.marker_positions = self.marker_states[:, 0:3]

        # self.dof_states = vec_dof_tensor
        # self.dof_positions = vec_dof_tensor[..., 0]
        # self.dof_velocities = vec_dof_tensor[..., 1]
        
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        # self.dof_positions = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 0]
        # self.dof_velocities = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)


        # Control tensor
        self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2))

        cam_pos = gymapi.Vec3(-1.0, -0.6, 0.8)
        cam_target = gymapi.Vec3(1.0, 1.0, 0.15)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Depth detection points. Origin is body origin(Can be identified in SolidWorks.)
        # exo_depth_points = heightmap_distribution( delta=0.1, limit=1.2,front_heavy=0.0, plot=False) #Uniform
        # exo_depth_points = heightmap_distribution( delta=0.07, limit=2,front_heavy=0.012, plot=False) # Weigted little towards front
        exo_dp_dim, exo_depth_points = heightmap_distribution( 1.12, 1.2, square=True, y_start=0.03, delta=0.05, front_heavy=0.0, plot=False) #Big square - Bounded by real bounds
        # exo_depth_points = heightmap_distribution( delta=0.06, limit=1.6,front_heavy=0.01, plot=True) # Weigted more towards front


        self.direction_vector = torch.zeros([self.num_envs, 2], device='cuda:0')
        # Convert numpy to tensor
        self.exo_depth_points_tensor = torch.tensor(exo_depth_points, device=self.device)
        # Initialize empty location tensor for all robots
        self.exo_locations_tensor = torch.zeros([self.num_envs, 6], device=self.device)
        # Tensor for heightmap memory
        self.heightmap_memory = torch.zeros(16,self.num_envs, self._num_camera_inputs,device=self.device) # save 16 timesteps -> 4 secs
        # Tensor that describes the valid heightmap entries
        self.exo_dp_overlay = heightmap_overlay(exo_dp_dim, self.exo_depth_points_tensor)

        # Spawn offset. Used for offsetting goal locations.
        self.spawn_offset = torch.zeros([self.num_envs, 3], device=self.device)
        
        # Move spawn points away from hills and rocks.
        self.check_spawn_slope(16)
        self.check_spawn_collision()
        self.check_spawn_slope(16)
        self.check_spawn_collision()

    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        
        #    - set up gravity
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81 
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
        # Terrain specifications
        terrain_width = 50 # terrain width [m]
        terrain_length = terrain_width # terrain length [m]
        self.terrain_side_length = terrain_width
        if terrain_length != terrain_width:
            print("!!!   terrain width != terrain height, PLEASE FIX   !!!")
        # KEEP TERRAIN WIDTH AND LENGTH EQUAL!!! - check_spawn_slope will not work if the are not.  
        horizontal_scale = 0.025#0.025
         # resolution per meter 
        vertical_scale = 0.005 # vertical resolution [m]
        self.heightfield = np.zeros((int(terrain_width/horizontal_scale), int(terrain_length/horizontal_scale)), dtype=np.int16)

        def new_sub_terrain(): return SubTerrain1(width=terrain_width,length=terrain_length,horizontal_scale=horizontal_scale,vertical_scale=vertical_scale)
        terrain = gaussian_terrain(new_sub_terrain(),0.5,0.0)
        #terrain.height_field_raw += np.random.rand(len(terrain.height_field_raw[0]),len(terrain.height_field_raw[1]))
        terrain = gaussian_terrain(terrain,15,5)
        #terrain = gaussian_terrain(terrain,5,1)
        #terrain = gaussian_terrain(terrain,1,0.4)
        #heightfield[0:int(terrain_width/horizontal_scale),:]= gaussian_terrain(new_sub_terrain()).height_field_raw
        rock_heigtfield, self.rock_positions = add_rocks_terrain(terrain=terrain)
        self.heightfield[0:int(terrain_width/horizontal_scale),:] = rock_heigtfield.height_field_raw
        vertices, triangles = convert_heightfield_to_trimesh1(self.heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=None)
        # Decimate mesh and reduce number of vertices
        vertices, triangles = polygon_reduction(vertices, triangles, target_vertices=200000)

        self.tensor_map = torch.tensor(self.heightfield, device='cuda:0')
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        tm_params = gymapi.TriangleMeshParams()

        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        # If the gound plane should be shifted:
        self.shift = -5
        tm_params.transform.p.x = self.shift
        tm_params.transform.p.y = self.shift
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)

    def check_spawn_collision(self):
        #self.initial_root_states[:,0] = torch.where(nearest_rock[:] <= 0.25,self.initial_root_states[:,0]+0.05,self.initial_root_states[:,0])
        for i in range(1,10000):
            self.exo_locations_tensor[:, 0:2] = self.initial_root_states[:,0:2].add(self.env_origins_tensor[:,0:2]) - self.shift
            dist_rocks = torch.cdist(self.exo_locations_tensor[:,0:2],self.rock_positions[:,0:2], p=2.0)   # Calculate distance to center of all rocks
            dist_rocks[:] = dist_rocks[:]-self.rock_positions[:,3]                               # Calculate distance to nearest point of all rocks
            nearest_rock = torch.min(dist_rocks,dim=1)[0]                                   # Find the closest rock to each robot
            self.initial_root_states[:,0] = torch.where(nearest_rock[:] <= 0.4,self.initial_root_states[:,0]+0.05,self.initial_root_states[:,0])


    def check_goal_collision(self,env_ids):
        ones = torch.ones_like(env_ids)
        zeros = torch.zeros_like(env_ids)
        global_pos = self.target_root_positions[env_ids,0:2].add(self.env_origins_tensor[env_ids,0:2]) - self.shift   # get target in global frame
        dist_rocks = torch.cdist(global_pos[:,0:2],self.rock_positions[:,0:2], p=2.0) # Calculate distance to center of all rocks
        dist_rocks[:] = dist_rocks[:]-self.rock_positions[:,3] # Calculate distance to nearest point of all rocks
        nearest_rock = torch.min(dist_rocks,dim=1)[0]   # Find the closest rock to each robot
        reset_buf = torch.where(nearest_rock <= self.rock_reset_distance_target, ones, zeros)  # Check if target is within reset distance
        env_ids = reset_buf * env_ids   # Multiply reset buffer with env_ids in order to get reset ids
        reset_buf_len = len(reset_buf.nonzero(as_tuple=False).squeeze(-1))  # Get number of non-zero values in the reset buffer
        return env_ids, reset_buf_len
      
    def check_spawn_slope(self, angle): 
        
        run_once = False
        offset_tensor = torch.zeros_like(self.initial_root_states[:, 0:2])

        for i in range(1,5000):
            
            offset = self.horizontal_scale # Move amount for each iteration - also determines the distance with which slope is measured.

            loc = self.initial_root_states[:, 0:2].add(self.env_origins_tensor[:, 0:2]) # Location of robots on map
            height_spawn = height_lookup(self.tensor_map, loc, self.horizontal_scale, self.vertical_scale, self.shift, loc, torch.zeros_like(self.exo_locations_tensor[:, 0:3]), self.exo_depth_points_tensor)
            
            # Slope in x direction
            loc_offset_x = loc
            loc_offset_x[:,0] += offset
            height_next_to_spawn_x = height_lookup(self.tensor_map, loc_offset_x, self.horizontal_scale, self.vertical_scale, self.shift, loc_offset_x, torch.zeros_like(self.exo_locations_tensor[:, 0:3]), self.exo_depth_points_tensor)
            
            # Slope in y direction
            loc_offset_y = loc
            loc_offset_y[:,1] += offset
            height_next_to_spawn_y = height_lookup(self.tensor_map, loc_offset_y, self.horizontal_scale, self.vertical_scale, self.shift, loc_offset_y, torch.zeros_like(self.exo_locations_tensor[:, 0:3]), self.exo_depth_points_tensor)
            
            # Slopes
            slope_x = height_spawn - height_next_to_spawn_x
            slope_y = height_spawn - height_next_to_spawn_y

            # Highest slope
            slope = torch.squeeze(torch.dstack((slope_x, slope_y)))
            slope = torch.amax(torch.abs(slope),1)            

            # Threshold for slope
            max_angle = math.sin(angle*0.0174533) #10 degrees
            slope_treshold = offset * max_angle
            offset_t = torch.ones_like(slope_x) * offset # Threshold as Tensor

            if run_once == False:
                # Fill with correct offset in direction of slope
                offset_tensor[:, 0] = torch.where(slope_x >= 0, offset_t, -offset_t)
                offset_tensor[:, 1] = torch.where(slope_y >= 0, offset_t, -offset_t)

                # 50 % chance of moving towards top of hill
                norm_dist = torch.randn(offset_tensor.shape[0], device='cuda:0')
                rand_factor = torch.copysign(torch.ones_like(offset_tensor[:,0]), norm_dist)
                offset_tensor *= rand_factor.unsqueeze(-1)

                run_once = True

            # Move if slope is high
            self.initial_root_states[:,0] = torch.where((torch.abs(slope) > slope_treshold), self.initial_root_states[:,0] + offset_tensor[:,0], self.initial_root_states[:,0])
            self.initial_root_states[:,1] = torch.where((torch.abs(slope) > slope_treshold), self.initial_root_states[:,1] + offset_tensor[:,1], self.initial_root_states[:,1])
            self.spawn_offset[:,0] = torch.where((torch.abs(slope) > slope_treshold), self.spawn_offset[:,0] + offset_tensor[:,0], self.spawn_offset[:,0])
            self.spawn_offset[:,1] = torch.where((torch.abs(slope) > slope_treshold), self.spawn_offset[:,1] + offset_tensor[:,1], self.spawn_offset[:,1])

            # Adjust offset if moved outside map
            offset_tensor[:, 0] = torch.where(self.initial_root_states[:,0].add(self.env_origins_tensor[:, 0]) > 0, offset_tensor[:,0], -offset_tensor[:,0])
            offset_tensor[:, 1] = torch.where(self.initial_root_states[:,1].add(self.env_origins_tensor[:, 1]) > 0, offset_tensor[:,1], -offset_tensor[:,1])
            offset_tensor[:, 0] = torch.where(self.initial_root_states[:,0].add(self.env_origins_tensor[:, 0]) < self.terrain_side_length+2*self.shift, offset_tensor[:,0], -offset_tensor[:,0])
            offset_tensor[:, 1] = torch.where(self.initial_root_states[:,1].add(self.env_origins_tensor[:, 1]) < self.terrain_side_length+2*self.shift, offset_tensor[:,1], -offset_tensor[:,1])

    def generate_goals(self,env_ids,radius=3):
        reset_buf_len = 1
        while (reset_buf_len > 0):
            self.random_goals(env_ids, radius=radius) # Generate random goals
            env_ids, reset_buf_len = self.check_goal_collision(env_ids) # Check if goals collides with random rocks

    def random_goals(self, env_ids, radius):
        num_sets = len(env_ids)
        alpha = 2 * math.pi * torch.rand(num_sets, device=self.device)
        TargetRadius = radius
        TargetCordx = 0
        TargetCordy = 0
        x = TargetRadius * torch.cos(alpha) + TargetCordx
        y = TargetRadius * torch.sin(alpha) + TargetCordy
        self.target_root_positions[env_ids, 0] = x + self.spawn_offset[env_ids, 0]
        self.target_root_positions[env_ids, 1] = y + self.spawn_offset[env_ids, 1]

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        
        self.generate_goals(env_ids, radius=3) # Generate goals

        self.marker_positions[env_ids] = self.target_root_positions[env_ids]
        
        global_pos = self.target_root_positions[env_ids, 0:2].add(self.env_origins_tensor[env_ids, 0:2])
        height_offset = height_lookup(self.tensor_map, global_pos, self.horizontal_scale, self.vertical_scale, self.shift, global_pos, torch.zeros(num_sets, 3), self.exo_depth_points_tensor)
        self.target_root_positions[env_ids, 2] = height_offset
        self.marker_positions[env_ids] = self.target_root_positions[env_ids] 
        actor_indices = self.all_actor_indices[env_ids, 1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,self.root_tensor, gymtorch.unwrap_tensor(actor_indices), num_sets)

        return actor_indices

    def _create_envs(self,num_envs,spacing, num_per_row):
       # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, 0.5 * -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, 0.5 * spacing, 0.0)

        asset_root = "../assets"
        exomy_asset_file = "urdf/exomy_modelv2/urdf/exomy_model.urdf"
        
        # if "asset" in self.cfg["env"]:f
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
            gymapi.DOF_MODE_VEL, #0  #L BOGIE
            gymapi.DOF_MODE_POS,  #1  #ML POS
            gymapi.DOF_MODE_VEL,  #2  #ML DRIVE
            gymapi.DOF_MODE_POS,  #3   #FL POS
            gymapi.DOF_MODE_VEL,  #4  #FL DRIVE
            gymapi.DOF_MODE_VEL, #5  #REAR BOGIE
            gymapi.DOF_MODE_POS,  #6  #RL POS
            gymapi.DOF_MODE_VEL,  #7  #RL DRIVE
            gymapi.DOF_MODE_POS,  #8  #RR POS
            gymapi.DOF_MODE_VEL,  #9  #RR DRIVE
            gymapi.DOF_MODE_VEL, #10 #R BOGIE
            gymapi.DOF_MODE_POS,  #11 #MR POS 
            gymapi.DOF_MODE_VEL,  #12 #MR DRIVE
            gymapi.DOF_MODE_POS,  #13 #FR POS
            gymapi.DOF_MODE_VEL,  #14 #FR DRIVE
        ]


        exomy_dof_props["friction"].fill(0.01)
        exomy_dof_props["stiffness"].fill(800)
        exomy_dof_props["damping"].fill(0.1)
        exomy_dof_props["armature"].fill(0.001)
        exomy_dof_props["effort"].fill(100)
        
        pose = gymapi.Transform()
        pose.p.z = 0.5
        # asset is rotated z-up by default, no additional rotations needed
        pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        self.exomy_handles = []
        self.envs = []
        env_origins = []

        #Create marker
        default_pose = gymapi.Transform()
        default_pose.p.z = 0.0
        default_pose.p.x = 0.1        
        marker_options = gymapi.AssetOptions()
        marker_options.fix_base_link = True
        marker_asset = self.gym.create_sphere(self.sim, 0.1, marker_options)
        for i in range(num_envs):
            # Create environment
            env0 = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env0)
            
            #Store environment origins
            origin = self.gym.get_env_origin(env0)
            env_origins.append([origin.x, origin.y, origin.z])

            exomy0_handle = self.gym.create_actor(
                env0,  # Environment Handle
                exomy_asset,  # Asset Handle
                pose,  # Transform of where the actor will be initially placed
                "exomy",  # Name of the actor
                i,  # Collision group that actor will be part of
                1,  # Bitwise filter for elements in the same collisionGroup to mask off collision
            )
            self.exomy_handles.append(exomy0_handle)
            self.gym.enable_actor_dof_force_sensors(env0, exomy0_handle)
            
            # Configure DOF properties
            # Set initial DOF states
            # gym.set_actor_dof_states(env0, exomy0_handle, default_dof_state, gymapi.STATE_ALL)
            # Set DOF control properties
            self.gym.set_actor_dof_properties(env0, exomy0_handle, exomy_dof_props)


            # Spawn marker
            marker_handle = self.gym.create_actor(env0, marker_asset, default_pose, "marker", i, 1, 1)
            self.gym.set_rigid_body_color(env0, marker_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))

        # Convert environment origins to tensor
        self.env_origins_tensor = torch.tensor(env_origins, device='cuda:0')

    def reset_idx(self, env_ids):
        #Used to reset multiple environments
        
        # set rotor speeds

        num_resets = len(env_ids)
        target_actor_indices = self.set_targets(env_ids)

        # Set orientation of robot as random around Z
        r = []
        actor_indices = self.all_actor_indices[env_ids, 0].flatten()
        for i in range(num_resets):
            r.append(R.from_euler('zyx', [(random.random() * 2 * math.pi), 0, 0], degrees=False).as_quat())

        RQuat = torch.cuda.FloatTensor(r)

        self.root_states[env_ids] = self.initial_root_states[env_ids]
        #self.root_states[env_ids, 0] = 0#torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        #self.root_states[env_ids, 1] = 0#torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 2] = 0.2#torch_rand_float(-0.2, 1.5, (num_resets, 1), self.device).flatten()

        #Sets orientation
        self.root_states[env_ids, 3:7] = RQuat
        
        # Spawn exomy at the correct z-height.
        loc = self.env_origins_tensor[env_ids] + self.initial_root_states[env_ids, 0:3]
        height = height_lookup(self.tensor_map, loc, self.horizontal_scale, self.vertical_scale, self.shift, loc, torch.zeros(num_resets, 3), self.exo_depth_points_tensor)
        self.root_states[env_ids, 2] = height+0.25

        self.dof_states = self.initial_dof_states
        self.gym.set_actor_root_state_tensor_indexed(self.sim,self.root_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets)
        #self.dof_positions = 0
        
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_states), gymtorch.unwrap_tensor(actor_indices), num_resets)

        self.last_dof_vel[env_ids] = 0.
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        return torch.unique(torch.cat([target_actor_indices, actor_indices]))

    def pre_physics_step(self, actions):

        target_actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
    
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        
        if len(reset_env_ids) > 0:
            actor_indices = self.reset_idx(reset_env_ids)

        reset_indices = torch.unique(torch.cat([target_actor_indices]))
        if len(reset_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(reset_indices), len(reset_indices))

        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        _actions = actions.to(self.device)

        '''
        # Code for running ExoMy in end-to-end mode
        actions_tensor[1::15]=(_actions[:,0]) * self.max_effort_pos  #1  #LF POS
        actions_tensor[2::15]=(_actions[:,1]) * self.max_effort_vel #2  #LF DRIVE
        actions_tensor[3::15]=(_actions[:,2]) * self.max_effort_pos #3  #LM POS
        actions_tensor[4::15]=(_actions[:,3]) * self.max_effort_vel #4  #LM DRIVE
        actions_tensor[6::15]=(_actions[:,4]) * self.max_effort_pos #6  #LR POS
        actions_tensor[7::15]=(_actions[:,5]) * self.max_effort_vel #7  #LR DRIVE
        actions_tensor[8::15]=(_actions[:,6]) * self.max_effort_pos #8  #RR POS
        actions_tensor[9::15]=(_actions[:,7]) * self.max_effort_vel #9  #RR DRIVE
        actions_tensor[11::15]=(_actions[:,8]) * self.max_effort_pos #11 #RF POS 
        actions_tensor[12::15]= (_actions[:,9]) * self.max_effort_vel #12 #RF DRIVE
        actions_tensor[13::15]=(_actions[:,10]) * self.max_effort_pos #13 #RM POS
        actions_tensor[14::15]=(_actions[:,11]) * self.max_effort_vel #14 #RM DRIVE
        '''
        
        # Code for running ExoMy in Ackermann mode
        _actions[:,0] = _actions[:,0] * 3
        _actions[:,1] = _actions[:,1] * 3
        
       # self.actions_driving_motors = torch.cat((self.actions_driving_motors[1:], motor_velocities.unsqueeze(dim=0)))
        #self.actions_steering_motors = torch.cat((self.actions_steering_motors[1:], steering_angles.unsqueeze(dim=0)))
        self.actions_nn = torch.cat((torch.reshape(_actions,(self.num_envs, self.cfg["env"]["numActions"], 1)), self.actions_nn), 2)[:,:,0:3]

        steering_angles, motor_velocities = Ackermann(_actions[:,0], _actions[:,1])
    
        # # a = torch.ones(1,device='cuda:0')*0.3
        # # b = torch.ones(1,device='cuda:0')*3
        # steering_angles, motor_velocities = Ackermann(a,b)

        steering_angles = -steering_angles
        
        actions_tensor[1::15]=(steering_angles[:,2])   #1  #ML POS
        actions_tensor[2::15]=(motor_velocities[:,2])  #2  #ML DRIVE
        actions_tensor[3::15]=(steering_angles[:,0])   #3   #FL POS
        actions_tensor[4::15]=(motor_velocities[:,0])  #4  #FL DRIVE
        actions_tensor[6::15]=(steering_angles[:,4])   #6  #RL POS
        actions_tensor[7::15]=(motor_velocities[:,4])  #7  #RL DRIVE
        actions_tensor[8::15]=(steering_angles[:,5])   #8  #RR POS
        actions_tensor[9::15]=(motor_velocities[:,5])  #9  #RR DRIVE
        actions_tensor[11::15]=(steering_angles[:,3])  #11 #MR POS  
        actions_tensor[12::15]=(motor_velocities[:,3]) #12 #MR DRIVE
        actions_tensor[13::15]=(steering_angles[:,1])  #13 #FR POS
        actions_tensor[14::15]=(motor_velocities[:,1]) #14 #FR DRIVE
        #



        
        '''
        # Code for extracting position and velocity goal over time.

        # Add new action and torques to "remember"-variable. Remove old action/torque. Used to compute rewards/penalties
            # Action
        self.actions_nn = torch.cat((torch.reshape(_actions,(self.num_envs, self.cfg["env"]["numActions"], 1)), self.actions_nn), 2)[:,:,0:3]
            # Steering angles
        steering_angles = torch.stack((steering_angles[:,1], steering_angles[:,3], steering_angles[:,5], steering_angles[:,0], steering_angles[:,2], steering_angles[:,4]), dim=1)
        #steering_angles = torch.stack()
        self.steering_angles = torch.cat((torch.reshape(steering_angles,(self.num_envs, 6, 1)), self.steering_angles), 2)[:,:,0:3]
            # Driving velocities
        driving_velocities = torch.stack((motor_velocities[:,1], motor_velocities[:,3], motor_velocities[:,5], motor_velocities[:,0], motor_velocities[:,2], motor_velocities[:,4]), dim=1)
        self.driving_velocities = torch.cat((torch.reshape(driving_velocities,(self.num_envs, 6, 1)), self.driving_velocities), 2)[:,:,0:3]
        '''

        '''
        # Code for manually setting the speed for exo motors.
        actions_tensor[1::15] = -math.pi/4 #1  #ML POS
        actions_tensor[2::15] = 0 #2  #ML DRIVE
        actions_tensor[3::15] = -math.pi/4 #3  #FL POS
        actions_tensor[4::15] = 0 #4  #FL DRIVE
        actions_tensor[6::15] = -math.pi/4 #6  #RL POS
        actions_tensor[7::15] = 0 #7  #RL DRIVE
        actions_tensor[8::15] = -math.pi/4 #8  #RR POS
        actions_tensor[9::15] = 0 #9  #RR DRIVE
        actions_tensor[11::15] = -math.pi/4 #11 #MR POS 
        actions_tensor[12::15] = 0 #12 #MR DRIVE
        actions_tensor[13::15] = -math.pi/4 #13 #FR POS
        actions_tensor[14::15] = 0 #14 #FR DRIVE
        speed =10
        actions_tensor[0] = 100 #BOTH REAR DRIVE        
        actions_tensor[2] = speed 
        actions_tensor[3] = 0
        actions_tensor[4] = speed 
        actions_tensor[7] = speed 
        actions_tensor[9] = speed 
        actions_tensor[12] = speed
        actions_tensor[14] = speed
        '''
        
        # Set 
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor)) #)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor)) #)
    #     start = self._num_observations
    #     end =  self._num_observations+self._num_camera_inputs
    #     new_heightmap_entry = self.obs_buf[...,start:end]
    #     new_heightmap_entry = new_heightmap_entry.unsqueeze(dim=0)
    #     self.heightmap_memory = self.push_to_tensor(self.heightmap_memory, new_heightmap_entry)

    # def push_to_tensor(self, tensor, x):
    #     return torch.cat((tensor[1:], x)) 

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        
        self.progress_buf += 1
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.root_euler = tensor_quat_to_eul(self.root_quats)

        # Compute location and rotation(RPY) for root body of each robot

        # TODO add offset to the global position
        self.exo_locations_tensor[:, 0:3] = self.location_tensor[:,0:3].add(self.env_origins_tensor)
        exo_rot = tensor_quat_to_eul(self.location_tensor[:,3:7])
        exo_rot[:,2] = torch.atan2(torch.sin(exo_rot[:,2]) * torch.cos(exo_rot[:,1]), torch.cos(exo_rot[:,2]) * torch.cos(exo_rot[:,0])) # Global direction
        
        # Compute depth point locations in x,y from robot orientation and location.
        depth_point_locations = exo_depth_observation(exo_rot, self.exo_locations_tensor[:,0:3], self.exo_depth_points_tensor)
        #print(depth_point_locations)
        # Lookup heigt at depth point locations.
        self.elevationMap = height_lookup(self.tensor_map, depth_point_locations, self.horizontal_scale, self.vertical_scale, self.shift, self.exo_locations_tensor[:,0:3], exo_rot, self.exo_depth_points_tensor)
        # Fill invalid intries with zeros
        # self.elevationMap[:] = self.elevationMap[:]*self.exo_dp_overlay    
        #print(torch.max(self.elevationMap[2]))
        # Visualize points for robot [0]
        #visualize_points(self.viewer, self.gym, self.envs[0], depth_point_locations[0, :, :], self.elevationMap[0:1,:], 0.1,self.exo_locations_tensor[:,0:3])
        
        # Calculate a direction vector for the robot
        self.direction_vector[:,0] = torch.cos(self.root_euler[..., 2] - (math.pi/2)) # x value
        self.direction_vector[:,1] = torch.sin(self.root_euler[..., 2] - (math.pi/2)) # y value
        self.target_vector = self.target_root_positions[..., 0:2] - self.root_positions[..., 0:2]
        self.compute_observations()
        self.compute_rewards()
        self.last_dof_vel[:] = self.dof_vel[:]

    def compute_observations(self):
        # eps = 1e-7
        #print(self.root_euler[0:10, 2])
        # dot = torch.sum(self.target_vector * self.direction_vector,dim=1) / (torch.linalg.norm(self.target_vector,dim=1) * torch.linalg.norm(self.direction_vector,dim=1))
        # angle = torch.clamp(dot, min = (-1 + eps), max = (1 - eps))
        # heading_diff = torch.arccos(angle)
        #https://stackoverflow.com/questions/2150050/finding-signed-angle-between-vectors
        heading_diff = torch.atan2(self.target_vector[:,0] * self.direction_vector[:,1] - self.target_vector[:,1]*self.direction_vector[:,0],self.target_vector[:,0]*self.direction_vector[:,0]+self.target_vector[:,1]*self.direction_vector[:,1])
        #dot =  ((target_vector[..., 0] * torch.cos(root_euler[..., 2] - (math.pi/2))) + (target_vector[..., 1] * torch.sin(root_euler[..., 2] - (math.pi/2)))) / ((torch.sqrt(torch.square(target_vector[..., 0]) + torch.square(target_vector[..., 1]))) * torch.sqrt(torch.square(torch.cos(root_euler[..., 2] - (math.pi/2))) + torch.square(torch.sin(root_euler[..., 2] - (math.pi/2)))))
        # angle = torch.clamp(dot, min = (-1 + eps), max = (1 - eps))
        # heading_diff = torch.arccos(angle)
        #pos_reward = 1.0 / (1.0 + target_dist * target_dist + (0.01 * progress_buf) + (0.5 * heading_diff))
        #self.obs_buf[..., 0:2] = (self.target_root_positions[..., 0:2] - self.root_positions[..., 0:2]) / 4
        self.obs_buf[..., 0] = torch.linalg.norm(self.target_vector,dim=1) / 4
        self.obs_buf[..., 1] = (heading_diff) / math.pi
        #self.obs_buf[..., 2] = (self.root_euler[..., 2])
        self.obs_buf[..., 2:4] = self.actions_nn[:,:,0] / 3
        self.obs_buf[...,self._num_observations:(self._num_observations+self._num_camera_inputs)] = self.elevationMap * 3
        # Prepare previous heightmap inputs
        # memories = torch.permute(self.heightmap_memory,(1,2,0)) # Shift dimensions to prepare for flattening
        # memories = memories[:,:,3::4] # take every fourth memory
        # memories = memories.flatten(1,2) # flatten input
        # self.obs_buf[..., self._num_camera_inputs+self._num_observations:self.cfg["env"]["numObservations"]]

        #print(self.elevationMap.shape)
        #print(torch.max(self.obs_buf[0,0:152]))
        #self.obs_buf[..., 3:6] = self.root_linvels
        #self.obs_buf[..., 6:9] = self.root_angvels
        # self.obs_buf[..., 0:3] = (self.target_root_positions - self.root_positions) / 3
        # self.obs_buf[..., 3:7] = self.root_quats
        # self.obs_buf[..., 7:10] = self.root_linvels / 2
        # self.obs_buf[..., 10:13] = self.root_angvels / math.pi
        return self.obs_buf

    def compute_rewards(self):
        # TODO remove shift from this formula
        
        self.exo_locations_tensor[:, 0:2] = self.exo_locations_tensor[:, 0:2] - self.shift
        self.rew_buf[:], self.reset_buf[:], extras = compute_exomy_reward(self.root_positions,
            self.target_root_positions, self.root_quats, self.root_euler, self.actions_nn,
            self.dof_force_tensor, self.exo_locations_tensor[:, 0:3], self.rock_positions,
            self.reset_buf, self.progress_buf, self.rew_scales, self.max_episode_length, self.dof_vel, self.last_dof_vel,
            self.rock_reset_distance_exomy, self.num_envs)
        self.extras["pos_reward"] = extras['pos_reward']
        self.extras["collision_penalty"] = extras['collision_penalty']
        self.extras["uprightness_penalty"] = extras['uprightness_penalty']
        self.extras["heading_contraint_penalty"] = extras['heading_contraint_penalty']
        self.extras["motion_contraint_penalty"] = extras['motion_contraint_penalty']
        self.extras["goal_angle_penalty"] = extras['goal_angle_penalty']
        self.extras["torque_penalty_driving"] = extras['torque_penalty_driving']
        self.extras["torque_penalty_steering"] = extras['torque_penalty_steering']      

@torch.jit.script
def compute_exomy_reward(root_positions, target_root_positions, 
        root_quats, root_euler, actions_nn, forces, global_location, rock_positions, reset_buf, progress_buf, rew_scales, max_episode_length, dof_vel, last_dof_vel,rock_reset_distance_exomy,num_envs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str,float], float, Tensor, Tensor, float, int) -> Tuple[Tensor, Tensor, Dict[str,Tensor]]

    # Tool tensors 
    zero_reward = torch.zeros_like(reset_buf)
    max_reward = torch.ones_like(reset_buf)
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    
    # Distance to target
    target_dist = torch.sqrt(torch.square(target_root_positions[..., 0:2] - root_positions[..., 0:2]).sum(-1))
    
    target_vector = target_root_positions[..., 0:2] - root_positions[..., 0:2]


    eps = 1e-7
    direction_vector = torch.zeros([ones.shape[0], 2], device='cuda:0')
    direction_vector[:,0] = torch.cos(root_euler[..., 2] - (math.pi/2)) # x value
    direction_vector[:,1] = torch.sin(root_euler[..., 2] - (math.pi/2)) # y value
    target_vector = target_root_positions[..., 0:2] - root_positions[..., 0:2]
    heading_diff = torch.atan2(target_vector[:,0] * direction_vector[:,1] - target_vector[:,1]*direction_vector[:,0],target_vector[:,0]*direction_vector[:,0]+target_vector[:,1]*direction_vector[:,1])



    # dot =  ((target_vector[..., 0] * torch.cos(root_euler[..., 2] - (math.pi/2))) + (target_vector[..., 1] * torch.sin(root_euler[..., 2] - (math.pi/2)))) / ((torch.sqrt(torch.square(target_vector[..., 0]) + torch.square(target_vector[..., 1]))) * torch.sqrt(torch.square(torch.cos(root_euler[..., 2] - (math.pi/2))) + torch.square(torch.sin(root_euler[..., 2] - (math.pi/2)))))
    # angle = torch.clamp(dot, min = (-1 + eps), max = (1 - eps))
    # heading_diff = torch.arccos(angle)
    # heading_diff_reward = torch.where(progress_buf < 200, (-heading_diff * (200-progress_buf))/200*(torch.abs(actions_nn[:,0,0]))*0.0, zero_reward)
    heading_diff_reward = zero_reward
    #pos_reward = 1.0 / (1.0 + target_dist * target_dist + (0.01 * progress_buf) + (0.5 * heading_diff))
    


    # Collision reward - Anton
    dist_rocks = torch.cdist(global_location[:,0:2],rock_positions[:,0:2], p=2.0)   # Calculate distance to center of all rocks
    dist_rocks[:] = dist_rocks[:]-rock_positions[:,3]                               # Calculate distance to nearest point of all rocks
    nearest_rock = torch.min(dist_rocks,dim=1)[0]                                   # Find the closest rock to each robot  
    collision_func = - 0.94/(1 + torch.square((nearest_rock-0.24)*5)+0.06)
    collision_penalty = torch.where(nearest_rock[:] < 1, collision_func ,zero_reward) * rew_scales['collision']
    # Collision rewardV2 - Anton
    
    # Uprightness 
    pitch = (((1)/(1+(torch.abs(root_euler[:,0])-0.78)**(2)))-0.73) * ((1)/(0.27))
    roll = (((1)/(1+(torch.abs(root_euler[:,1])-0.78)**(2)))-0.73) * ((1)/(0.27))
    pitch_reward = torch.where(torch.abs(root_euler[:,0]) > 0.1745, pitch, zero_reward)
    roll_reward = torch.where(torch.abs(root_euler[:,1]) > 0.1745, roll, zero_reward)
    uprightness_penalty = -((pitch_reward + roll_reward) / 2) * rew_scales["uprightness"]

    # Heading constraint - Ikke køre baglæns
    lin_vel = actions_nn[:,0,0]    # Get latest lin_vel
    heading_contraint_penalty = torch.where(lin_vel < 0, -max_reward, zero_reward) * rew_scales["heading"]


    # Motion constraint - Ikke oscilere på output
    #motion_contraint_penalty = -torch.abs(actions_nn[:,0,0] - actions_nn[:,0,1]) * rew_scales["motion_contraint"]
    # v2
    penalty1 = torch.where((torch.abs(actions_nn[:,0,0] - actions_nn[:,0,1]) > 0.05), torch.square(torch.abs(actions_nn[:,0,0] - actions_nn[:,0,1])),zero_reward)
    penalty2 = torch.where((torch.abs(actions_nn[:,1,0] - actions_nn[:,1,1]) > 0.05), torch.square(torch.abs(actions_nn[:,1,0] - actions_nn[:,1,1])),zero_reward)
    motion_contraint_penalty =  torch.pow(penalty1,2) * rew_scales["motion_contraint"]
    motion_contraint_penalty = motion_contraint_penalty+(torch.pow(penalty2,2)) * rew_scales["motion_contraint"]
    
    # accereration contraint
    acceleration_penalty = torch.sum(torch.square(last_dof_vel - dof_vel), dim=1) * rew_scales['acceleration']

    # Torque reward
    driving_forces = torch.stack((forces[2::15], forces[4::15], forces[7::15], forces[9::15], forces[12::15], forces[14::15]), dim=1)
    steering_forces = torch.stack((forces[1::15], forces[3::15], forces[6::15], forces[8::15], forces[11::15], forces[13::15]), dim=1)

    torque_penalty_driving = -torch.abs(driving_forces).sum(dim=1) * rew_scales['torque_driving']
    torque_penalty_steering = -torch.abs(steering_forces).sum(dim=1) * rew_scales['torque_steering']


    # Reward closer to target
    close_to_target_penalty = -1/(1+20*torch.pow(target_dist+0.2,5))  * torch.abs(actions_nn[:,0,0]) * rew_scales['vel_near_goal']
    close_to_target_penalty += -1/(1+20*torch.pow(target_dist+0.2,5))  * torch.abs(actions_nn[:,1,0]) * rew_scales['vel_near_goal']


    # Total distance
    # rew_scales["total_distance"]
    #REMOVE
    torque_penalty_driving = close_to_target_penalty
    
    #Normalized penalty
    

    #goal_angle_penalty = (heading_diff/3.1415) * rew_scales['goal_angle']
    goal_angle_penalty = torch.where(torch.abs(heading_diff) > 2, -torch.abs(heading_diff*0.3*rew_scales['goal_angle']), zero_reward)


    # distance to target
    pos_reward = (1.0 / (1.0 + target_dist * target_dist)) * rew_scales['pos']
    pos_reward = torch.where(target_dist <= 0.03, 1.03*(max_episode_length-progress_buf), pos_reward)  # reward for getting close to target
    # Total reward
    reward = pos_reward + collision_penalty + uprightness_penalty + heading_contraint_penalty + motion_contraint_penalty + torque_penalty_driving + torque_penalty_steering + close_to_target_penalty + heading_diff_reward + goal_angle_penalty
    # normalized reward
    

    # resets due to episode length'
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)
    reset = torch.where(target_dist >= 4, ones, reset)
    reset = torch.where(nearest_rock <= rock_reset_distance_exomy, ones, reset)  # reset if colliding
    reward = torch.where(nearest_rock <= rock_reset_distance_exomy, reward-100, reward)  # reset if colliding
    reset = torch.where(target_dist <= 0.03, ones, reset)  # reset if close to target

    reset = torch.where(torch.abs(root_euler[:,0]) >= 0.78*1.5, ones, reset)  # reset if roll above 45 degrees(radians)
    reset = torch.where(torch.abs(root_euler[:,1]) >= 0.78*1.5, ones, reset)  # reset if pitch above 45 degrees(radians)
    
    # Track number of collision 
    collision_tracker = torch.where(nearest_rock <= rock_reset_distance_exomy, max_reward*num_envs, zero_reward)  # Track number of collisions
    reward = reward / 3000
    extras = {}

    extras['pos_reward'] = pos_reward 
    extras['collision_penalty'] = collision_tracker#collision_penalty
    extras['goal_angle_penalty'] = goal_angle_penalty
    extras['uprightness_penalty'] = uprightness_penalty
    extras['heading_contraint_penalty'] = heading_contraint_penalty
    extras['motion_contraint_penalty'] = motion_contraint_penalty
    extras['torque_penalty_driving'] = torque_penalty_driving
    extras['torque_penalty_steering'] = torque_penalty_steering
    extras['close_to_target_penalty'] = close_to_target_penalty
    extras['acceleration_penalty'] = acceleration_penalty
    #extras['torque_penalty_steering'] = torque_penalty_steering
    
    return reward, reset, extras       

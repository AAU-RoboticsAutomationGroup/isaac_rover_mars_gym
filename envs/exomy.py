"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
"""


import math
from click import option
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
from isaacgym.terrain_utils import *



def print_asset_info(asset, name):
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print("Got %d bodies, %d joints, and %d DOFs" % (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))

    # Iterate through joints
    print("Joints:")
    for i in range(num_joints):
        name = gym.get_asset_joint_name(asset, i)
        type = gym.get_asset_joint_type(asset, i)
        type_name = gym.get_joint_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

    # iterate through degrees of freedom (DOFs)
    print("DOFs:")
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        type = gym.get_asset_dof_type(asset, i)
        type_name = gym.get_dof_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))


#################################################
# initialize gym
gym = gymapi.acquire_gym()

#################################################
# Add custom arguments
custom_parameters = [
    {
        "name": "--num_envs",
        "type": int,
        "default": 16,
        "help": "Number of environments to create",
    },
]
# parse arguments
args = gymutil.parse_arguments(
    description="Joint control Methods Example",
    custom_parameters=custom_parameters,
)

#################################################
# create a simulator
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -3.721) #Define gravity

sim_params.substeps = 4
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.75
sim_params.flex.warm_start = 0.8
sim_params.use_gpu_pipeline = False

if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

    
#################################################
# Allocates which device will simulate and which device will
# render the scene. Defines the simulation type to be used
#################################################
sim = gym.create_sim(
    args.compute_device_id,  # Index of CUDA-enabled GPU to be used for simulation
    args.graphics_device_id,  # Index of GPU to be used for rendering
    gymapi.SIM_FLEX,  # FORCING FLEX SIMULATION FOR PROPPER COLLISION MESHES
    sim_params,  # Simulation parameters
)

if sim is None:
    print("*** Failed to create sim")
    quit()

#################################################
# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError("*** Failed to create viewer")


"""
Randomly generate a terrain height map and translate it to a mesh
"""
#################################################
# Randomly generate a terrain height map and translate it to a mesh (Not used)
terrain_width = 120.
terrain_length = 120.
horizontal_scale = 0.25  # [m]
vertical_scale = 0.005  # [m]
num_rows = int(terrain_width/horizontal_scale)
num_cols = int(terrain_length/horizontal_scale)
#heightfield = np.zeros((1*num_rows, num_cols), dtype=np.int16)

slope = random_uniform_terrain((SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)), min_height=-0.15, max_height=0.15, step=0.02, downsampled_scale=0.5).height_field_raw

vertices2, triangles2 = convert_heightfield_to_trimesh(slope, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices2.shape[0]
tm_params.nb_triangles = triangles2.shape[0]
tm_params.transform.p.x = -50.
tm_params.transform.p.y = -50.





#################################################
# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

#################################################
# add ground plane (Not used)
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
#gym.add_ground(sim, plane_params)

#################################################
# set up the env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, 0.0)
print("Creating %d environments" % num_envs)

#################################################
# LOAD ASSETS
#################################################
# add exomy urdf asset
asset_root = "../assets"
exomy_asset_file = "urdf/exomy_model/urdf/exomy_model.urdf"




################################################
# add World environment asset
world_asset_file = "urdf/exomy_model/World.urdf"

world_options = gymapi.AssetOptions()
world_options.disable_gravity = False 
world_options.fix_base_link = True #Fix world in place
world_options.armature = 0.01
world_options.convex_decomposition_from_submeshes = True
#world_options.flip_visual_attachments = False
world_options.vhacd_enabled = False

#################################################
# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.disable_gravity = False
asset_options.armature = 0.01
# use default convex decomposition params
asset_options.vhacd_enabled = False


#################################################
#Set position and rotation of the world
world_pose = gymapi.Transform()
world_pose.p = gymapi.Vec3(0, 0, -1)
world_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)

#################################################
# asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (exomy_asset_file, asset_root))
exomy_asset = gym.load_asset(sim, asset_root, exomy_asset_file, asset_options)

#################################################
# Load world asset
print("Loading asset '%s' from '%s'" % (world_asset_file, asset_root))
world_asset = gym.load_asset(sim, asset_root, world_asset_file, world_options)



#################################################
# get joint limits and ranges for Franka
exomy_dof_props = gym.get_asset_dof_properties(exomy_asset)
exomy_lower_limits = exomy_dof_props["lower"]
exomy_upper_limits = exomy_dof_props["upper"]
exomy_ranges = exomy_upper_limits - exomy_lower_limits
exomy_mids = 0.5 * (exomy_upper_limits + exomy_lower_limits)
exomy_num_dofs = len(exomy_dof_props)

#################################################
# set default DOF states
default_dof_state = np.zeros(exomy_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = exomy_mids

#################################################
# Set all DOF to the same driveMode.
# NOTE! For ExoMy the DOF (non-fixed joints) are controlled by position (STR) and velocity (DRV).
# Bogie joints are passive (no motors): LFB, RFB, MRB
# Check ExoMy_asset_info.txt for the order of DOF
# exomy_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
#    DOF_MODE_NONE lets the joints move freely within their range of motion
#    DOF_MODE_POS
#    DOF_MODE_VEL
#    DOF_MODE_EFFORT
# set DOF control properties
#print(exomy_dof_props)
exomy_dof_props["driveMode"] = (
    gymapi.DOF_MODE_NONE,  # LFB bogie
    gymapi.DOF_MODE_POS,
    gymapi.DOF_MODE_VEL,
    gymapi.DOF_MODE_POS,
    gymapi.DOF_MODE_VEL,
    gymapi.DOF_MODE_NONE,  # MRB bogie
    gymapi.DOF_MODE_POS,
    gymapi.DOF_MODE_VEL,
    gymapi.DOF_MODE_POS,
    gymapi.DOF_MODE_VEL,
    gymapi.DOF_MODE_NONE,  # RFB bogie
    gymapi.DOF_MODE_POS,
    gymapi.DOF_MODE_VEL,
    gymapi.DOF_MODE_POS,
    gymapi.DOF_MODE_VEL,
    gymapi.DOF_MODE_POS,  # Left eye
    gymapi.DOF_MODE_POS,  # Right eye
)

exomy_dof_props["stiffness"].fill(800.0)
exomy_dof_props["damping"].fill(40.0)

#################################################
# initial root pose for exomy actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, 0.27)
initial_pose.r = gymapi.Quat(0, 0.0, 1.0, 0.0)


#################################################
# Set up environments
envs = []
#gym.add_triangle_mesh(sim, vertices2.flatten(), triangles2.flatten(), tm_params)

#################################################
# Make a environment for the world, -1 menas it collides with everything, allowing the robots to drive on it
envWorld = gym.create_env(sim, env_lower, env_upper, num_per_row)
# Spawn the world in the environment
world_handle = gym.create_actor(envWorld, world_asset, world_pose, "world", -1, 0)



#################################################
# Make environment for every Exomy actor and spawn it at the desired location
for i in range(num_envs):
    # Create environment
    env0 = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env0)
    tm_params.transform.p.x = initial_pose.p.x - 1.
    tm_params.transform.p.y = initial_pose.p.y -1.

    
    exomy0_handle = gym.create_actor(
        env0,  # Environment Handle
        exomy_asset,  # Asset Handle
        initial_pose,  # Transform of where the actor will be initially placed
        "exomy",  # Name of the actor
        i,  # Collision group that actor will be part of
        1,  # Bitwise filter for elements in the same collisionGroup to mask off collision
    )

    
    
    # Configure DOF properties
    # Set initial DOF states
    # gym.set_actor_dof_states(env0, exomy0_handle, default_dof_state, gymapi.STATE_ALL)
    # Set DOF control properties
    gym.set_actor_dof_properties(env0, exomy0_handle, exomy_dof_props)

    # Move the different motors (DOF)
    # Set DOF drive 'position' targets
    STR_LF_joint_dof_handle = gym.find_actor_dof_handle(env0, exomy0_handle, "STR_LF_joint")
    STR_LM_joint_dof_handle = gym.find_actor_dof_handle(env0, exomy0_handle, "STR_LM_joint")
    STR_LR_joint_dof_handle = gym.find_actor_dof_handle(env0, exomy0_handle, "STR_LR_joint")
    STR_RF_joint_dof_handle = gym.find_actor_dof_handle(env0, exomy0_handle, "STR_RF_joint")
    STR_RM_joint_dof_handle = gym.find_actor_dof_handle(env0, exomy0_handle, "STR_RM_joint")
    STR_RR_joint_dof_handle = gym.find_actor_dof_handle(env0, exomy0_handle, "STR_RR_joint")
    gym.set_dof_target_position(env0, STR_LF_joint_dof_handle, math.radians(-55))
    gym.set_dof_target_position(env0, STR_LM_joint_dof_handle, math.radians(0))
    gym.set_dof_target_position(env0, STR_LR_joint_dof_handle, math.radians(55))
    gym.set_dof_target_position(env0, STR_RF_joint_dof_handle, math.radians(55))
    gym.set_dof_target_position(env0, STR_RM_joint_dof_handle, math.radians(0))
    gym.set_dof_target_position(env0, STR_RR_joint_dof_handle, math.radians(-55))

    # Set DOF drive 'velocity' targets
    DRV_LF_joint_dof_handle = gym.find_actor_dof_handle(env0, exomy0_handle, "DRV_LF_joint")
    DRV_LM_joint_dof_handle = gym.find_actor_dof_handle(env0, exomy0_handle, "DRV_LM_joint")
    DRV_LR_joint_dof_handle = gym.find_actor_dof_handle(env0, exomy0_handle, "DRV_LR_joint")
    DRV_RF_joint_dof_handle = gym.find_actor_dof_handle(env0, exomy0_handle, "DRV_RF_joint")
    DRV_RM_joint_dof_handle = gym.find_actor_dof_handle(env0, exomy0_handle, "DRV_RM_joint")
    DRV_RR_joint_dof_handle = gym.find_actor_dof_handle(env0, exomy0_handle, "DRV_RR_joint")
    gym.set_dof_target_velocity(env0, DRV_LF_joint_dof_handle, -0.5 * math.pi)
    gym.set_dof_target_velocity(env0, DRV_LM_joint_dof_handle, -0.5 * math.pi)
    gym.set_dof_target_velocity(env0, DRV_LR_joint_dof_handle, -0.5 * math.pi)
    gym.set_dof_target_velocity(env0, DRV_RF_joint_dof_handle, 0.5 * math.pi)
    gym.set_dof_target_velocity(env0, DRV_RM_joint_dof_handle, 0.5 * math.pi)
    gym.set_dof_target_velocity(env0, DRV_RR_joint_dof_handle, 0.5 * math.pi)


#################################################
# Configure camera to look at the first env
cam_pos = gymapi.Vec3(-1.0, -0.6, 0.8)
cam_target = gymapi.Vec3(1.0, 1.0, 0.15)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

#################################################
# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

#################################################
# print_asset_info(exomy_asset,"ExoMy")
#################################################


#################################################
# Simulate
#################################################
while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Nothing to be done for env 0

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

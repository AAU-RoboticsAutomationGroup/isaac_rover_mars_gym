from ctypes import sizeof
import torch
from isaacgym import gymapi
import numpy as np

def exo_depth_observation(exo_r, exo_l, exo_depth_points):
    # X:0, Y:1, Z:2

    # Get number of points and number of robots from input
    num_points = exo_depth_points.size()[0]
    num_robots = exo_r.size()[0]

    # Expand depth point vectors to be martix of size[1, x](from vector of size[x])
    x = exo_depth_points[:,0].expand(1, num_points)
    y = exo_depth_points[:,1].expand(1, num_points)

    # Compute sin and cos to all angles
    sinxr = torch.transpose(torch.sin(exo_r[:,0].expand(1, num_robots)), 0, 1)
    cosxr = torch.transpose(torch.cos(exo_r[:,0].expand(1, num_robots)), 0, 1)
    sinyr = torch.transpose(torch.sin(exo_r[:,1].expand(1, num_robots)), 0, 1)
    cosyr = torch.transpose(torch.cos(exo_r[:,1].expand(1, num_robots)), 0, 1)
    sinzr = torch.transpose(torch.sin(exo_r[:,2].expand(1, num_robots)), 0, 1)
    coszr = torch.transpose(torch.cos(exo_r[:,2].expand(1, num_robots)), 0, 1)

    # Expand location vector to be of size[x, y], from size[x]
    exo_xl = torch.transpose(exo_l[:, 0].expand(num_points,num_robots), 0, 1)
    exo_yl = torch.transpose(exo_l[:, 1].expand(num_points,num_robots), 0, 1)

    # Add local point offsets to origin of each robot. Scale offset as function of rotation around x and y
    x_p = exo_xl + coszr * x * torch.abs(cosyr) - sinzr * y * torch.abs(cosxr)
    y_p = exo_yl + sinzr * x * torch.abs(cosyr) + coszr * y * torch.abs(cosxr)

    #Stack points in a [x, y, 2] matrix, and return
    return torch.stack((x_p, y_p), 2)

def height_lookup(heightmap: torch.Tensor, depth_points: torch.Tensor, horizontal_scale, vertical_scale, shift, exo_loc):

    # Scale locations to fit heightmap
    scaledmap = (depth_points-shift)/horizontal_scale
    # Bound values inside the map
    scaledmap = torch.clamp(scaledmap, min = 0, max = heightmap.size()[0]-1)
    # Round to nearest integer
    scaledmap = torch.round(scaledmap)

    # If more than two dimensions in scaledmap(depth_points are multiple x, y coordinates for each robot)
    if scaledmap.dim() > 2:
        # Convert x,y coordinates to two vectors.
        x = scaledmap[:,:,0]
        y = scaledmap[:,:,1]
        x = x.reshape([(depth_points.size()[0]* depth_points.size()[1]), 1])
        y = y.reshape([(depth_points.size()[0]* depth_points.size()[1]), 1])
        x = x.type(torch.long)
        y = y.type(torch.long)

        # Lookup heights in heightmap
        heights = heightmap[x, y]
        # Scale to fit actual height, dependent on resolution
        heights = heights * vertical_scale
        # Reshape heigts to correct size
        heights = heights.reshape([depth_points.size()[0], depth_points.size()[1]])

        # Set visualize variable
        visualize = True
        
        # If vizualisation not on, shift Z-coordinates to correct values
        if visualize == False:
            # Z-shift, so points are measured relative to robot height.
            exo_z_loc = torch.transpose(exo_loc[:,2].expand(depth_points.size()[1], depth_points.size()[0]), 0, 1)
            heights = heights - exo_z_loc
        else:
            # Else, print message.
            print("!!! DO NOT TRAIN ON THIS. HEIGHTS ARE OFFSET. !!!")

    # If 2 or less dimensions in scaled map(1 point for each robot) - Used to spawn robot at correct height
    else :

        # Convert x,y coordinates to two vectors.
        x = scaledmap[:,0]
        y = scaledmap[:,1]
        x = x.type(torch.long)
        y = y.type(torch.long)

        # Lookup heights in heightmap
        heights = heightmap[x, y]
        
        # Scale to fit actual height, dependent on resolution
        heights = heights * vertical_scale

    # Return the found heights
    return heights

def visualize_points(viewer, gym, environment, points, heights, line_length):

    # Find number of points to visualize
    lines_amount = points.size()[0]

    # Convert points and heigts to numpy array
    points = points.cpu().numpy()
    heights = heights.cpu().numpy().transpose()
    
    # Store poins and heights in single variable
    lines = np.concatenate((points, heights), axis=1)

    # Arrays for storring converted information about lines
    verts = np.empty((lines_amount, 2), gymapi.Vec3.dtype)
    colors = np.empty(lines_amount, gymapi.Vec3.dtype)

    for i, a in enumerate(lines[:,0]):
        # Origin of line
        verts[i][0] = (lines[i, 0], lines[i, 1], lines[i, 2])
        # End coordinate for line
        verts[i][1] = (lines[i, 0], lines[i, 1], lines[i, 2]+line_length)
        # Color for line(Normalized RGB)
        colors[i] = (1, 0, 0)

    # Delete previously drawn lines
    gym.clear_lines(viewer)
    # Draw new lines
    gym.add_lines(viewer, environment, lines_amount, verts, colors)

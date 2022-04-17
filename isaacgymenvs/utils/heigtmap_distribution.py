import dis
from importlib_metadata import distribution
import numpy as np
#import heigtmap_distribution
import matplotlib.pyplot as plt

def heightmap_distribution( delta=0, front_heavy=0, limit=3, plot=True):

    point_distribution = []

    # If delta variable not set, exit.
    if delta == 0:
        print("Need delta value!")
        exit()

    y = 0.296
    while y < limit:
        
        x = 0

        delta += front_heavy

        while x < limit_at_x(y):
            
            if x == 0:
                point_distribution.append([-x, -y])
            else:
                point_distribution.append([-x, -y])
                point_distribution.append([x, -y])
            
            x += delta

        y += delta

    point_distribution = np.round(point_distribution, 4)
    
    #print("Distribution created:")
    #print(np.shape(point_distribution))
    #print(distribution)

    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(point_distribution[:,0], point_distribution[:,1])
        ax.set_aspect('equal')
        plt.show()

    return point_distribution

def limit_at_x(x):
    return x*(0.24555/0.296)+0.13338

if __name__ == "__main__":
    heightmap_distribution( delta=0.06, limit=1.6,front_heavy=0.01, plot=True)
import dis
from importlib_metadata import distribution
import numpy as np
#import heigtmap_distribution
import matplotlib.pyplot as plt

def heightmap_distribution(x_limit, y_limit, square=False, y_start=0.296, delta=0, front_heavy=0, plot=True):

    point_distribution = []

    # If delta variable not set, exit.
    if delta == 0:
        print("Need delta value!")
        exit()

    y = y_start
    while y < y_limit:
        
        x = 0

        delta += front_heavy

        flag = True
        %
        if square==False:
            limit = limit_at_x(y)
        else:
            limit = x_limit


        while x < limit:
            
            if x < -limit:
                x += delta
                flag = False

            if flag:
                x -= delta
            else:
                point_distribution.append([x, -y])
                x += delta

        y += delta

    point_distribution = np.round(point_distribution, 4)


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
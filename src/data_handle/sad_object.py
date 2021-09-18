import sys

import math, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection

from PIL import Image

import pyclipper


def return_Map():
    '''10m x 10m area'''
    boundary_coords = [(0.1, 0.1), (9.9, 0.1), (9.9, 9.9), (0.1, 9.9)]
    obstacle_list = [ 
        [(0.0, 0.0), (0.0, 10.0), (2.0, 10.0), (2.0, 0.0)],
        [(8.0, 0.0), (8.0, 10.0), (10.0, 10.0), (10.0, 0.0)],
        [(4.0, 4.0), (4.0, 6.0), (6.0, 6.0), (6.0, 4.0)],
     ]

    return boundary_coords, obstacle_list

def get_ref_path(path=1):
    if path == 1:
        start = (5, 1)
        turning1 = (5, 3)
        turning2 = (3, 4)
        turning3 = (3, 6)
        turning4 = (5, 7)
        end = (5, 9)
    elif path == 2:
        start = (5, 1)
        turning1 = (5, 3)
        turning2 = (7, 4)
        turning3 = (7, 6)
        turning4 = (5, 7)
        end = (5, 9)
    else:
        sys.exit(0)
    return [start, turning1, turning2, turning3, turning4, end]

def plot_path(path, ax, color='k--'):
    for i in range(len(path)-1):
        ax.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], color)

class MovingObject():
    def __init__(self, current_position, stagger=0):
        self.stagger = stagger
        self.traj = [current_position]

    def motion_model(self, ts, state, action):
        x,y = state[0], state[1]
        vx, vy = action[0], action[1]
        x += ts*vx
        y += ts*vy
        return (x,y)

    def one_step(self, ts, action):
        self.traj.append(self.motion_model(ts, self.traj[-1], action))

    def run(self, path, ts=.2, vmax=0.5):
        whole_path  = path
        coming_path = whole_path[1:]
        while(len(coming_path)>0):
            stagger = random.randint(0,10)/10*self.stagger
            x, y = self.traj[-1][0], self.traj[-1][1]
            dist_to_next_goal = math.hypot(coming_path[0][0]-x, coming_path[0][1]-y)
            if dist_to_next_goal < (vmax*ts):
                coming_path.pop(0)
                continue
            else:
                if random.randint(0,1)>0.5:
                    stagger = -stagger
                dire = ((coming_path[0][0]-x)/dist_to_next_goal, (coming_path[0][1]-y)/dist_to_next_goal)
                action = (dire[0]*math.sqrt(vmax)+stagger, dire[1]*math.sqrt(vmax)+stagger)
                self.one_step(ts, action)


class Graph:
    def __init__(self, boundary_coords, obstacle_list, inflation=0):
        self.boundary_coords = boundary_coords # in counter-clockwise ordering
        self.obstacle_list = obstacle_list.copy() # in clock-wise ordering
        self.preprocess_obstacle_list = obstacle_list.copy()
        self.inflator = pyclipper.PyclipperOffset()
        self.preprocess_obstacle_list[2] = self.preprocess_obstacle(pyclipper.scale_to_clipper(obstacle_list[2]), pyclipper.scale_to_clipper(inflation))

    def preprocess_obstacle(self, obstacle, inflation):
        self.inflator.Clear()
        self.inflator.AddPath(obstacle, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        inflated_obstacle = pyclipper.scale_from_clipper(self.inflator.Execute(inflation))[0]
        inflated_obstacle.reverse()
        return inflated_obstacle

    def plot_map(self, ax, clean=False, empty=False):
        boundary = self.boundary_coords + [self.boundary_coords[0]]
        boundary = np.array(boundary)
        ax.set_xlim([min(boundary[:,0]), max(boundary[:,0])])
        ax.set_ylim([min(boundary[:,1]), max(boundary[:,1])])
        if empty:
            ax.plot(boundary[:,0], boundary[:,1], 'white')
            ax.axis('equal')
            return
        for obs in self.obstacle_list:
            obs = np.array(obs)
            poly = patches.Polygon(obs, color='gray')
            ax.add_patch(poly)
        if not clean:
            ax.plot(boundary[:,0], boundary[:,1], 'k')
            for obs in self.preprocess_obstacle_list:
                obs_edge = obs + [obs[0]]
                xs, ys = zip(*obs_edge)
                ax.plot(xs,ys,'b')
        ax.axis('equal')


if __name__ == '__main__':

    inflation = 0
    stagger = 0.3
    start = get_ref_path()[0]
    ts = 0.2
    vmax = 1

    ref_path1 = get_ref_path(1)
    ref_path2 = get_ref_path(2)
    boundary_coords, obstacle_list = return_Map()
    graph = Graph(boundary_coords, obstacle_list, inflation=inflation)

    obj = MovingObject(start, stagger)
    obj.run(ref_path1, ts, vmax)
    traj = obj.traj

    fig, ax = plt.subplots()
    # ------------------------
    ax.axis('off')
    ax.margins(0)
    graph.plot_map(ax, clean=False, empty=False)
    plot_path(ref_path1, ax, color='rx--')
    ax.plot(np.array(traj)[:,0],np.array(traj)[:,1],'k.')
    ax.axis('equal')
    plt.tight_layout()
    # ------------------------
    plt.show()
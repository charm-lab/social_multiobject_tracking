from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shapely.affinity
import numpy as np
from multi_tracking import frame_scaling_factor
import constants
from bisect import bisect_left
from scipy.interpolate import interp1d

def mmToIndex(val):
    return round(frame_scaling_factor*val*(1.0/25.4))
def indexTomm(val):
    return (1.0/frame_scaling_factor)*val*25.4

class Tactor:
    def __init__(self, point, weight, row, col):
        self.point = point #mm
        self.weight = weight
        self.row = row
        self.col = col
        self.traj_locs = {} #mm
        self.actuator_locs = {} #mm
        self.main_path_locs = {}
        self.actuator_relative_mm = [] #mm
        self.touch_mean = 0
        self.touch_std = 0
        self.neighbors = set([])
        self.scale = 1
        self.velocity_scaled_locs = []

    def setRadius(self, radius):
        self.radius = radius

    def applyTransformation(self, transform_set):
        x_trans, y_trans, scale, rotation, vert_trans = transform_set
        self.point = shapely.affinity.rotate(self.point, rotation, origin=(0,0), use_radians=True)
        self.point = shapely.affinity.scale(self.point, xfact = scale, yfact = scale, origin=(0,0))
        self.point = shapely.affinity.translate(self.point, xoff=x_trans, yoff=y_trans + vert_trans*self.row)
        self.scale = scale

    def getCopy(self):
        return Tactor(Point(self.point), self.weight, self.row, self.col)

    def addTrajLoc(self, time, x, y, pressure):
        self.traj_locs[time] = [x, y, pressure]

    def clearTrajLocs(self):
        self.traj_locs = {}


    def getAllActuations(self, frames):
        self.actuator_locs = {}
        self.main_path_locs = {}
        neighbor_going = False
        neighbor_time = None
        for time in range(len(frames)):
            if time in self.traj_locs:
                x = self.point.x+self.traj_locs[time][0]
                y = self.point.y+self.traj_locs[time][1]
                z = self.traj_locs[time][2]
                self.actuator_locs[time] = [x,y,z]
                self.main_path_locs[time] = [x,y,z]
            else:
                neighbor_found = False
                for neighbor in self.neighbors:
                    if time in neighbor.traj_locs:
                        [max_x, max_y] = self.getMaxLoc(frames[time])
                        if max_x is None:
                            continue
                        z = frames[time][max_y, max_x]
                        neighbor_z = neighbor.traj_locs[time][2]
                        neighbor_found = True
                        if not neighbor_going:
                            neighbor_time = time
                            neighbor_going = True
                        self.actuator_locs[time] = [indexTomm(max_x), indexTomm(max_y), z]
                        neighbor_x = neighbor.point.x + neighbor.traj_locs[time][0]
                        neighbor_y = neighbor.point.y + neighbor.traj_locs[time][1]
                        neighbor_z = neighbor.traj_locs[time][2]
                        self.main_path_locs[time] = [neighbor_x, neighbor_y, neighbor_z]

    def getActuatorRelativemm(self, frames):
        out_locs = []
        for time in range(len(frames)):
            out_locs.append([0,0,0])
        for time in self.actuator_locs.keys():
            cur_loc = [0,0,0]
            #print(time, self.actuator_locs[time][1], self.actuator_locs[time][0])
            cur_loc[0] = self.actuator_locs[time][0] - self.point.x
            cur_loc[1] = self.actuator_locs[time][1] - self.point.y
            cur_loc[2] = self.actuator_locs[time][2]- self.touch_mean
            out_locs[time] = cur_loc
        self.actuator_relative_mm = out_locs



    def addNeighbor(self, tactor):
        self.neighbors.add(tactor)

    def getMaxLoc(self, frame):
        maxVal = -float("inf")
        maxLoc = [None,None]
        rows = np.shape(frame)[0]
        cols = np.shape(frame)[1]
        for row in range(rows):
            for col in range(cols):
                if self.coordsInTactor(col, row):
                    if frame[row][col] > maxVal:
                        maxVal = frame[row][col]
                        maxLoc = [col, row]
        return maxLoc


    def coordsInTactor(self, x, y):
        x = x*(1.0/frame_scaling_factor)*25.4
        y = y*(1.0/frame_scaling_factor)*25.4
        relative_x = x - self.point.x
        relative_y = y - self.point.y
        dist_squared = relative_x**2 + relative_y**2
        if dist_squared <= self.radius**2:
            return True
        else:
            return False

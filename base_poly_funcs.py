from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

center_dist_width = 37
center_dist_height = 50

virtual_radius = center_dist_width/2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shapely.affinity
import numpy as np
from Tactor import *
#frame_scaling = 3

def generateAlignedCircles():
    tactors = []
    for h in range(2):
        for w in range(4):
            weight = 1
            if w == 1:
                weight = 1.02
            tactors.append(Tactor(Point(center_dist_width*w, center_dist_height*h), weight, h, w, np.array([center_dist_width*w, center_dist_height*h])))
    return tactors

def generateNeighborSet(tactors):
    for base_tactor in tactors:
        for neighbor_candidate in tactors:
            if base_tactor.row == neighbor_candidate.row and (np.abs(base_tactor.col - neighbor_candidate.col) == 1 or np.abs(base_tactor.col - neighbor_candidate.col) == 1):
                base_tactor.addNeighbor(neighbor_candidate)

def getTranslationVals(frame_width_height):
    x_translations = np.linspace(0-virtual_radius, frame_width_height[0]-virtual_radius*6, num=20)
    y_translations = np.linspace(0, frame_width_height[1], num=20)
    #x_translations = [100]
    #y_translations = [60]
    return x_translations, y_translations

def getScalingVals():
    return [1.5,1.6,1.7]
    #return np.linspace(1, 2, num=5)

def getRotationVals():
    rotations = [0,1,-1,2,-2,3,-3]
    rotations = [rot*np.pi/(3*24) for rot in rotations]
    rotations = [0]
    return rotations

def applyTransformations(tactors, transform_set):
    transformed_tactors = []
    for tactor in tactors:
        new_tactor = tactor.getCopy()
        new_tactor.applyTransformation(transform_set)
        transformed_tactors.append(new_tactor)
    return transformed_tactors

def generateTransformSets(frame_width_height, scalings = None):
    x_translations, y_translations = getTranslationVals(frame_width_height)
    scalings = getScalingVals()
    scalings = [1]
    rotations = [0]
    transform_sets = []
    for scale in scalings:
        for x_trans in x_translations:
            for y_index in range(len(y_translations)):
                y_trans = y_translations[y_index]
                for rotation in rotations:
                    vert_splits = np.linspace(0, frame_width_height[1] - y_trans, num=len(y_translations)-y_index, endpoint=True)
                    for vert_split in vert_splits:
                        transform_sets.append([x_trans,y_trans, scale, rotation, vert_split])
    return transform_sets


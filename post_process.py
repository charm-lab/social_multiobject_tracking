import numpy as np
import scipy
import math
import os
from skimage.feature import peak_local_max
from collections import Counter
tare_count = 5

def cropAndFlipFrames(frames):
    test_frames = np.sum(frames, axis=0)
    indices = np.argwhere(~np.isnan(test_frames))
    minrow = np.min(indices[:,0])
    maxrow = np.max(indices[:,0])
    mincol = np.min(indices[:, 1])
    maxcol = np.max(indices[:, 1])
    cropped_frames = frames[:, minrow:maxrow+1, mincol:maxcol+1]
    if mincol < 30:
        #flip so upper arm is always on left. 
        cropped_frames = np.flip(cropped_frames, 2)
    return cropped_frames

# remove buffer data
def tareAndRemoveEdges(frames, buffer):
    if buffer > 0:
        frames = frames - np.mean(frames[0:buffer], axis=0)
    return frames[buffer:len(frames) - buffer]

def getLocalMaxima(frames, threshold_rel = .5, min_distance = 4, exclude_border = 0):
    bool_frames = np.zeros(np.shape(frames))
    # if a cell has 4 or fewer neighbors, never let it be a maximum.
    bad_cells = findBadCells(frames)
    for i in range(len(frames)):
        no_nan_frame = np.nan_to_num(frames[i])
        bool_frames[i,:,:] = peak_local_max(no_nan_frame, min_distance = min_distance, threshold_rel=threshold_rel, exclude_border = exclude_border, indices=False)
    bool_frames[:,bad_cells[0], bad_cells[1]] = False
    return bool_frames

def findBadCells(frames):
    bad_is = []
    bad_js = []
    testFrame = frames[0]
    width = np.shape(testFrame)[1]
    height = np.shape(testFrame)[0]
    for i in range(height):
        for j in range(width):
            valid_sum = 0
            for i_surround in range(-1, 2):
                for j_surround in range(-1,2):
                    cur_i = i + i_surround
                    cur_j = j + j_surround
                    if ( (not (i_surround == 0 and j_surround == 0)) and cur_i >= 0 and cur_i <  height \
                        and cur_j >= 0 and cur_j < width and not np.isnan(testFrame[cur_i,cur_j])):
                        valid_sum += 1
            if (valid_sum < 5):
                bad_is.append(i)
                bad_js.append(j)
    return [bad_is, bad_js]

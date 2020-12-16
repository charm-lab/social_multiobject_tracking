import dependencies.mcf.build.python_lib.mcf as mcf
from post_process import getLocalMaxima
from scipy.stats import norm
from general_plotting import *
import numpy as np
import random
import itertools
import cvxpy as cvx
import scipy.stats
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from base_poly_funcs import *
from joblib import Parallel, delayed
import multiprocessing
import copy
from Tactor import *
import constants
import scipy.io as sio
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import multiprocessing
import time
#female lower arm start pixel: 10
#male lower arm start pixel: 12
# use 8mm circle
#

 #this means each cell is 1/3in
num_cores = multiprocessing.cpu_count()
def enlargeFrames(frames, factor = 3):
    frames = np.kron(frames, np.ones((1, factor, factor)))
    return frames

def blurFrame(frame, sigmas):
    out_frame = scipy.ndimage.filters.gaussian_filter(frame, sigmas)
    return out_frame

def getOnlyBaseTrajectories(frames):
    frames = np.nan_to_num(frames);
    frames = enlargeFrames(frames, factor = frame_scaling_factor)
    for i in range(np.shape(frames)[0]):
        frames[i] = blurFrame(frames[i], [2, 2])
    whole_mean = np.mean(frames)
    whole_std = np.std(frames)
    base_trajectories = generateTrajectories(frames, whole_mean, whole_std, use_maxima=True)
    return [base_trajectories, [], [], []]

def generateOptimalTrajectories(frames, scalings = None, raw_trajectories = False, ignore_circles = False):
    frames = np.nan_to_num(frames);
    frames = enlargeFrames(frames, factor = frame_scaling_factor)
    for i in range(np.shape(frames)[0]):
        frames[i] = blurFrame(frames[i], [blurring, blurring])
    whole_mean = np.mean(frames)
    whole_std = np.std(frames)
    base_trajectories = generateTrajectories(frames, whole_mean, whole_std, source_sink_cost = source_sink_cost, use_maxima=True, max_dist = max_dist, std_dev = std_dev)
    
    frame_width_mm = np.shape(frames[0])[1]*25.4/frame_scaling_factor
    frame_height_mm = np.shape(frames[0])[0]*25.4/frame_scaling_factor
    frame_width_height = (frame_width_mm, frame_height_mm)
    transform_sets = generateTransformSets(frame_width_height, scalings = scalings)

    aligned_tactors = generateAlignedCircles()
    best_trajectory_set = None
    best_trajectory_set_cost = float("inf")
    best_transform = None
    all_trajectory_sets = []
    frame_length = len(frames)
    subset_lengths = 16

    transform_subsets = []
    for i in range(int(len(transform_sets)/subset_lengths) + 1):
        if i*subset_lengths + subset_lengths < len(transform_sets):
            transform_subset = transform_sets[i*subset_lengths:i*subset_lengths + subset_lengths]
        else:
            transform_subset = transform_sets[i*subset_lengths:]
        transform_subsets.append(transform_subset)
    base_kernel = aligned_tactors
    num_cores = multiprocessing.cpu_count()
    # this line is testing every trajectory
    trajectory_set_costs = Parallel(n_jobs = num_cores-1)(delayed(pruneBaseTrajectoriesCosts)(base_trajectories, frame_length, base_kernel, transform_subsets[i]) for i in range(len(transform_subsets)))
    trajectory_set_costs = [cost for subset in trajectory_set_costs for cost in subset]
    for i in range(len(trajectory_set_costs)):
        if trajectory_set_costs[i] < best_trajectory_set_cost:
            best_trajectory_index = i
            best_trajectory_set_cost = trajectory_set_costs[i]
    # this line is just repulling the best trajectory
    best_trajectory_set = pruneBaseTrajectories(base_trajectories, frame_length, base_kernel, transform = transform_sets[best_trajectory_index])
    best_transform = transform_sets[best_trajectory_index]  
    best_trajectory_set[1] = frames

    tactors = best_trajectory_set[2]
    for tactor in tactors:
        tactor.touch_mean = whole_mean
        tactor.touch_std = whole_std
    setTactorLocations(best_trajectory_set)

    return best_trajectory_set

def getUpsampledTrajFrames(trajectory_set):
    out_trajectories, frames, tactors, cur_virtual_radius, trajectory_set_cost = trajectory_set
    upsampled_frames = np.repeat(frames, 10, axis=0)
    for trajectory in out_trajectories:
        times = trajectory['times']
        ten_times = [time*10 for time in times]
        new_times = list(range(min(ten_times), max(ten_times)))
        new_times.append(max(ten_times))
        rows = trajectory['rows']
        cols = trajectory['cols']
        x_mm = trajectory['x_mm']
        y_mm = trajectory['y_mm']
        pressures = trajectory['pressures']
        f_rows = interp1d(ten_times, rows)
        f_cols = interp1d(ten_times, cols)
        f_x_mm = interp1d(ten_times, x_mm)
        f_y_mm = interp1d(ten_times, y_mm)
        f_pressures = interp1d(ten_times, pressures)
        new_rows = f_rows(new_times)
        new_cols = f_cols(new_times)
        new_x_mm = f_x_mm(new_times)
        new_y_mm = f_y_mm(new_times)
        new_pressures = f_pressures(new_times)
        trajectory['rows'] = new_rows
        trajectory['cols'] = new_cols
        trajectory['x_mm'] = new_x_mm
        trajectory['y_mm'] = new_y_mm
        trajectory['pressures'] = new_pressures
        trajectory['times'] = new_times
    upsampled_trajectory_set = (out_trajectories, upsampled_frames, tactors, cur_virtual_radius, trajectory_set_cost)
    setTactorLocations(upsampled_trajectory_set, clear_locs = True)
    return upsampled_trajectory_set

def setTactorLocations(trajectory_set, clear_locs = False):
    out_trajectories, frames, tactors, cur_virtual_radius, trajectory_set_cost = trajectory_set
    for tactor in tactors:
        tactor.clearTrajLocs()
    for trajectory in out_trajectories:
        for tactor in tactors:
            for time in range(len(frames)):
                if trajectoryInPoly(trajectory, tactor, time, cur_virtual_radius):
                    index = trajectory['times'].index(time)
                    point = Point(trajectory['x_mm'][index], trajectory['y_mm'][index])
                    relative_x = point.x - tactor.point.x
                    relative_y = point.y - tactor.point.y
                    tactor.addTrajLoc(time, relative_x, relative_y, trajectory['pressures'][index])
    for tactor in tactors:
        tactor.getAllActuations(frames)
        tactor.getActuatorRelativemm(frames)


def pruneBaseTrajectories(base_trajectories, num_time_indices, tactors, transform = None):
    trajectory_set_cost = 0
    cur_trajectories = copy.deepcopy(base_trajectories)
    if transform is not None:
        x_trans, y_trans, scale, rotation, vert_trans = transform
        tactors = applyTransformations(tactors, transform)
        generateNeighborSet(tactors)
        cur_virtual_radius = scale*virtual_radius
        for tactor in tactors:
            tactor.setRadius(cur_virtual_radius)
    time_vertex_boxes = findTrajectoryBounders(cur_trajectories, tactors, num_time_indices, cur_virtual_radius)

    invalid_sets = findInvalidSets(time_vertex_boxes)
    out_trajectories = pruneTrajectories(cur_trajectories, invalid_sets, num_time_indices)
    for out_trajectory in out_trajectories:
        out_trajectory['in_poly_costs'] = [out_trajectory['costs'][i]*out_trajectory['in_poly_weight'][i] for i in range(len(out_trajectory['costs']))]
        out_trajectory['in_poly_total_cost'] = sum(out_trajectory['in_poly_costs'])
        trajectory_set_cost += out_trajectory['in_poly_total_cost']
    return [out_trajectories, num_time_indices, tactors, cur_virtual_radius, trajectory_set_cost]

def pruneBaseTrajectoriesCosts(base_trajectories, num_time_indices, tactors, transform_set = None):
    trajectory_set_costs = []
    for transform in transform_set:
        trajectory_set_cost = 0
        cur_trajectories = copy.deepcopy(base_trajectories)
        if transform is not None:
            x_trans, y_trans, scale, rotation, vert_trans = transform
            temp_tactors = applyTransformations(tactors, transform)
            generateNeighborSet(temp_tactors)
            cur_virtual_radius = scale*virtual_radius
            for tactor in temp_tactors:
                tactor.setRadius(cur_virtual_radius)
        time_vertex_boxes = findTrajectoryBounders(cur_trajectories, temp_tactors, num_time_indices, cur_virtual_radius)

        invalid_sets = findInvalidSets(time_vertex_boxes)
        out_trajectories = pruneTrajectories(cur_trajectories, invalid_sets, num_time_indices)
        for out_trajectory in out_trajectories:
            out_trajectory['in_poly_costs'] = [out_trajectory['costs'][i]*out_trajectory['in_poly_weight'][i] for i in range(len(out_trajectory['costs']))]
            out_trajectory['in_poly_total_cost'] = sum(out_trajectory['in_poly_costs'])
            trajectory_set_cost += out_trajectory['in_poly_total_cost']
        trajectory_set_costs.append(trajectory_set_cost)
    return trajectory_set_costs

def generateTrajectories(frames, whole_mean, whole_std, source_sink_cost = 20, use_maxima = True, max_dist = 50, std_dev = 1.25):
    if use_maxima:
        bool_frames = getLocalMaxima(frames, threshold_rel = 0, min_distance = 1, exclude_border = 0);
        masked_frames = bool_frames*frames;
    else:
        masked_frames = frames
    max_val = np.max(masked_frames)
    min_val = np.min(masked_frames)
    mean_val = np.mean(frames)
    virtual_max = max_val - min_val
    g = mcf.Graph();
    nodes_time_row_col = [()]; #node names start at 1, so we create empty val
    pair_costs = {}
    nodes_by_frame = {}
    for i in range(len(masked_frames)):
        (maxima_rows, maxima_cols) = np.where(masked_frames[i] > .00001)
        nodes_by_frame[i] = []
        for j in range(len(maxima_rows)):
            val = masked_frames[i, maxima_rows[j], maxima_cols[j]] - min_val;
            #probability = 1-norm.cdf(val, gaussian_mean, gaussian_stddev)*.9 
            std_devs_away = (masked_frames[i, maxima_rows[j], maxima_cols[j]] - whole_mean)/whole_std
            probability = 1-(scipy.stats.norm.cdf(std_devs_away-std_dev))*.98 # call one std dev away chance
            node_val = np.log(probability/(1-probability));
            new_node = g.add(node_val);
            nodes_time_row_col.append((i, node_val, maxima_rows[j], maxima_cols[j]))
            nodes_by_frame[i].append((new_node, maxima_rows[j], maxima_cols[j], val))
            g.link(g.ST, new_node, source_sink_cost)
            g.link(new_node, g.ST, source_sink_cost)
            if i > 0:
                for (prev_node, prev_row, prev_col, prev_val) in nodes_by_frame[i-1]:
                    xy_dist_squared = (maxima_rows[j] - prev_row)**2 +(maxima_cols[j] - prev_col)**2
                    if xy_dist_squared < max_dist:
                        dist_squared = xy_dist_squared
                        if prev_node not in pair_costs.keys():
                            pair_costs[prev_node] = {}
                        xy_cost = -np.log(1-dist_squared/max_dist)
                        pair_costs[prev_node][new_node] = xy_cost
                        g.link(prev_node, new_node, xy_cost)
    trajectories = mcf.Solver(g).run_search(1, 200);
    #return dictionary with all trajectories
    index = 0;
    num_time_indices = len(frames)
    trajectory_set_cost = 0
    out_trajectories = []
    for trajectory in trajectories:
        cur_trajectory = {}
        cur_trajectory['times'] = [];
        cur_trajectory['costs'] = [];
        cur_trajectory['rows'] = [];
        cur_trajectory['cols'] = [];
        cur_trajectory['in_poly_costs'] = [];
        cur_trajectory['x_mm'] = [];
        cur_trajectory['y_mm'] = [];
        cur_trajectory['pressures'] = [];
        cur_trajectory['transition_costs'] = [];
        cur_trajectory['index'] = index;
        cur_trajectory['touch_length'] = len(masked_frames)
        cur_trajectory['total_cost'] = 0;
        cur_trajectory['tactors'] = [[] for i in range(len(masked_frames))]
        cur_trajectory['in_poly_weight'] = []
        for loc_index in range(len(trajectory)):
            loc = trajectory[loc_index]
            if loc_index > 0:
                prev_loc = trajectory[loc_index - 1]
                cur_trajectory['transition_costs'].append(pair_costs[prev_loc][loc])
            time, cost, row, col = nodes_time_row_col[loc]
            cur_trajectory['times'].append(time)
            cur_trajectory['costs'].append(cost)
            cur_trajectory['rows'].append(row)
            cur_trajectory['cols'].append(col)
            cur_trajectory['in_poly_weight'].append(0)
            cur_trajectory['in_poly_costs'].append(cost)
            row_range = 5
            col_range = 5
            if False and row >= row_range and row < np.shape(frames)[1]-row_range and col >= col_range and col < np.shape(frames)[2]-col_range:
                pressure_window = frames[time, row-row_range:row+row_range+1, col-col_range:col+col_range+1]
                (new_row, new_col) = scipy.ndimage.measurements.center_of_mass(pressure_window - np.amin(pressure_window))
                new_row += row-1 + .5
                new_col += col-1 + .5
            else:
                new_row = row + .5
                new_col = col + .5
            cur_trajectory['x_mm'].append(new_col*(1.0/frame_scaling_factor)*25.4)
            cur_trajectory['y_mm'].append(new_row*(1.0/frame_scaling_factor)*25.4)
            cur_trajectory['pressures'].append(frames[time, row, col])
        cur_trajectory['total_cost'] = sum(cur_trajectory['costs'])
        cur_trajectory['in_poly_total_cost'] = sum(cur_trajectory['costs'])
        trajectory_set_cost += cur_trajectory['total_cost']
        index+=1;
        out_trajectories.append(cur_trajectory)
    return out_trajectories

def findInvalidSets(time_vertex_boxes):
    invalid_sets = []
    for time in range(len(time_vertex_boxes)):
        vertex_boxes = time_vertex_boxes[time]
        # these are trajectory indices
        nonempty_indices = [index for index in range(len(vertex_boxes)) if len(vertex_boxes[index]) > 0]

        for i in range(len(nonempty_indices)+1):
            #all combinations of i nonempty_indices
            combinations = itertools.combinations(nonempty_indices, i)
            for combination in list(combinations):
                # if we already know this set is bad EVER, skip
                if set(combination) in invalid_sets:
                    continue
                current_test = [vertex_boxes[i] for i in combination]
                if not bipartiteCover(current_test):
                    invalid_sets.append(set(combination))
    return invalid_sets

#http://pages.cs.wisc.edu/~shuchi/courses/787-F09/scribe-notes/lec5.pdf
def bipartiteCover(bounding_box_sets):
    for i in range(1, len(bounding_box_sets)+1):
        combinations = itertools.combinations(bounding_box_sets, i)
        for combination in combinations:
            combined_set = set([])
            # box_set is actually the tactors

            for box_set in combination:
                combined_set = combined_set | box_set
            if len(combined_set) < i:
                return False
    return True

 
def findTrajectoryBounders(trajectories, tactors, num_indices, virtual_radius):
    if len(trajectories) == 0:
        return trajectories
    time_vertex_boxes = []
    for time in range(num_indices):
        time_vertex_boxes.append([])
        for trajectory_index in range(len(trajectories)):
            time_vertex_boxes[time].append(set([]))
            trajectory = trajectories[trajectory_index]
            for tactor_index in range(len(tactors)):
                tactor = tactors[tactor_index]
                if trajectoryInPoly(trajectory, tactor, time, virtual_radius):
                    index = trajectory['times'].index(time)
                    point = Point(trajectory['x_mm'][index], trajectory['y_mm'][index])
                    relative_x = point.x - tactor.point.x
                    relative_y = point.y - tactor.point.y
                    dist_squared = relative_x**2 + relative_y**2
                    trajectory['in_poly_weight'][index] = tactor.weight*(1.04 - .04*dist_squared/(virtual_radius**2))
                    time_vertex_boxes[time][trajectory_index].add(tactor_index)
                    trajectory['tactors'][time].append(tactor_index)
    return time_vertex_boxes

def pruneTrajectories(trajectories, invalid_sets, num_indices):
    optimal_trajectories = findOptimalTrajectories(trajectories, invalid_sets)
    out_trajectories = []
    for optimal_trajectory in optimal_trajectories:
        if np.sum(optimal_trajectory['in_poly_weight']) > 0.000001:
            out_trajectories.append(optimal_trajectory)
    return out_trajectories


def trajectoryInPoly(trajectory, tactor, time, virtual_radius):
    #tactor dims are in mm.
    if time in trajectory['times']:
        index = trajectory['times'].index(time)
        point = Point(trajectory['x_mm'][index], trajectory['y_mm'][index])
        relative_x = point.x - tactor.point.x
        relative_y = point.y - tactor.point.y
        dist_squared = relative_x**2 + relative_y**2
        if dist_squared <= virtual_radius**2:
            return True
    return False



def findOptimalTrajectories(trajectories, invalid_sets):
    if len(trajectories) == 0:
        print("No trajectories in optimization")
        return []
    v = cvx.Variable(len(trajectories), boolean=True)
    weights = [-trajectory['total_cost'] for trajectory in trajectories]
    objective = cvx.Maximize(v@weights)
    constraints = []
    trajes = []
    for traj_indices in invalid_sets:
        # can't have all these trajectories, so must have at most n-1 of them.
        variable_list = [v[traj_index] for traj_index in traj_indices]
        constraints.append(cvx.sum(variable_list) <= len(variable_list)-1)
    prob = cvx.Problem(objective, constraints)
    solution = prob.solve(solver=cvx.ECOS_BB)
    if solution is None or v.value is None:
        print("Bad solution")
        return []
    else:
        return [trajectories[index] for index in range(len(trajectories)) if v.value[index] > .99]

def getTrajectoryHighlights(trajectories, frames, tactors, cur_virtual_radius, ignore_circles = False):
    highlights = {}
    height = np.shape(frames)[0]
    for i in range(len(frames)):
        highlights[i] = []
    traj_num = 0;
    traj_colors = {};
    out_polys = []
    if not ignore_circles:
        if tactors is not None:
            for tactor in tactors:
                points = list(tactor.point.buffer(cur_virtual_radius).exterior.coords)
                recovered_points = [[frame_scaling_factor*point[0]*(1.0/25.4), frame_scaling_factor*point[1]*(1.0/25.4)] for point in points[0:len(points)-1]]
                out_polys.append(recovered_points)
        # generate colors to plot
        for tactor in tactors:
            for time in tactor.actuator_locs.keys():
                row = frame_scaling_factor*tactor.actuator_locs[time][1]*(1.0/25.4)
                col = frame_scaling_factor*tactor.actuator_locs[time][0]*(1.0/25.4)
                if time not in tactor.traj_locs.keys():
                    highlights[time].append({'row': row, 'col': col, 'color': (1,1,1)})
                else:
                    highlights[time].append({'row': row, 'col': col, 'color': (0,0,0)})
    if ignore_circles:
        for trajectory in trajectories:
          for index in range(len(trajectory['times'])):
              time = trajectory['times'][index]
              row = trajectory['rows'][index]
              col = trajectory['cols'][index]
              if traj_num not in traj_colors.keys():
                  traj_colors[traj_num] = (random.random(),random.random(),random.random());
              highlights[time].append({'row': row, 'col': col, 'color': traj_colors[traj_num]})
          traj_num = traj_num + 1;
    return highlights, out_polys

def makeTrajectoryPlots(cur_touch_trajectories, data_instance, gif_name, save_dir, save_suffix, ignore_circles = False):
    touch_trajectories = cur_touch_trajectories[0]
    cur_frames = cur_touch_trajectories[1]
    tactors = cur_touch_trajectories[2]
    cur_virtual_radius = cur_touch_trajectories[3]
    highlights, out_polys = getTrajectoryHighlights(touch_trajectories, cur_frames, tactors, cur_virtual_radius, ignore_circles)
    title = ""
    title += str(data_instance['emotion_instance']) + '_' + str(data_instance['subject_number']) + '_' + str(data_instance['emotion']) + "_Multi"
    plotGif(cur_frames, save_dir + "/", gif_name + save_suffix, title = title, highlights = highlights, polygons = out_polys, frequency=1)

def generateMatFiles(cur_touch_trajectories, touches_data_index, data_index, save_dir, save_suffix):
    touch_trajectories = cur_touch_trajectories[0]
    frames = cur_touch_trajectories[1]
    tactors = cur_touch_trajectories[2]
    cur_virtual_radius = cur_touch_trajectories[3]
    out_mat = np.zeros([len(touch_trajectories), 3*len(frames)])
    for i in range(len(touch_trajectories)):
        trajectory = touch_trajectories[i]
        for index in range(len(trajectory['times'])):
            time = trajectory['times'][index]
            y_mm = trajectory['y_mm'][index]
            x_mm = trajectory['x_mm'][index]
            row = trajectory['rows'][index]
            col = trajectory['cols'][index]
            pressure = trajectory['pressures'][index]
            out_mat[i, time*3 + 0] = x_mm
            out_mat[i, time*3 + 1] = y_mm
            out_mat[i, time*3 + 2] = pressure
    os.makedirs(save_dir + "/", exist_ok=True)
    sio.savemat(save_dir + "/" + str(data_index) + save_suffix + '.mat', {'trajectories' : out_mat})

def generateVoicecoilFiles(cur_touch_trajectories, touches_data_index, data_index, save_dir, save_suffix):
    touch_trajectories = cur_touch_trajectories[0]
    frames = cur_touch_trajectories[1]
    tactors = cur_touch_trajectories[2]
    cur_virtual_radius = cur_touch_trajectories[3]
    out_mat = np.zeros([len(frames), len(tactors)*3])
    out_array = []
    max_val = -float("inf")
    min_val = float("inf")


    for i in range(min(len(frames), touches_data_index["orig_length"])):
        for j in [2,1,0,5,4,3]:
            tactor = tactors[j]
            cur_triplet = tactor.actuator_relative_mm[i]
            if cur_triplet[0] == 0 and cur_triplet[1] == 0 and cur_triplet[2] == 0:
                out_array.append(None)
            elif cur_triplet[0] > tactor.radius or cur_triplet[0] < -tactor.radius \
                or cur_triplet[1] > tactor.radius or cur_triplet[1] < -tactor.radius:
                out_array.append(None)
            else:
                cur_val = tactor.actuator_relative_mm[i][2]
                if cur_val > max_val:
                    max_val = cur_val
                if cur_val < min_val:
                    min_val = cur_val
                out_array.append(cur_val) 
    # normalize between 0 and 1
    for i in range(len(out_array)):
        if out_array[i] is None:
            out_array[i] = min_val
    out_array = [(val-min_val)/(max_val - min_val) for val in out_array]
    out_array = [str(val) for val in out_array]
    out_string = " ".join(out_array)
    text_file = open(save_dir + "/" + str(data_index) + save_suffix, 'w')
    text_file.write(out_string)
    text_file.close()
    os.makedirs(save_dir + touches_data_index['emotion'] + "/", exist_ok=True)
    sio.savemat(save_dir + touches_data_index['emotion'] + "/" + str(data_index) + save_suffix + '.mat', {'tactor_locs' : out_mat})

def plotTrajectoryIntensities(cur_touch_trajectories, touches_data_index, data_index, save_dir, save_suffix):
    trajectories = cur_touch_trajectories[0]
    frames = cur_touch_trajectories[1]
    press_scaling = 30
    plt.figure()
    for trajectory in trajectories:
        num_times = len(trajectory['times'])
        x_outs = []
        out_min_press = []
        out_max_press = []
        times = []
        for i in range(len(trajectory['times'])):
            time = trajectory['times'][i]
            cur_press = trajectory['pressures'][i]
            cur_x = trajectory['x_mm'][i]
            times.append(time)
            x_outs.append(cur_x)
            out_min_press.append(time - cur_press*press_scaling)
            out_max_press.append(time + cur_press*press_scaling)
        plt.plot(x_outs, times)
        plt.plot(x_outs, out_min_press)
        plt.plot(x_outs, out_max_press)
        for i in range(len(x_outs)-1):
            plt.fill_between(x_outs[i:i+2], out_min_press[i:i+2], out_max_press[i:i+2], color = 'green', alpha = .5)
    plt.show()


def plotSignalIntensities(cur_touch_trajectories, image_dir, title_name):
    trajectories = cur_touch_trajectories[0]
    frames = cur_touch_trajectories[1]
    smoothing_window = 3
    plt.figure()
    for trajectory in trajectories:
        num_times = len(trajectory['times'])
        times = []
        pressures = []
        fake_distances = []
        for i in range(len(trajectory['times'])):
            time = trajectory['times'][i]
            cur_press = trajectory['pressures'][i]
            cur_x = trajectory['x_mm'][i]
            cur_y = trajectory['y_mm'][i]
            times.append(time)
            pressures.append(cur_press)
        plt.plot([t/20 for t in times], pressures)
        plt.xlabel('time (s)')
        plt.ylabel('pressure (psi)')
    plt.savefig(image_dir + title_name + '_intensities.png', bbox_inches='tight')

import sys
import os
import pickle
from post_process import *
from multi_tracking import *
from generate_voicecoils import generateVCOuput
import math



buffer = 20

def loadData(data_file):
    if os.path.isfile(data_file):
        print("Data file found, loading")
        joint_subject_data = pickle.load(open(data_file, "rb"))
        print("Data loaded")
    else:
        print("Data file not found")
        joint_subject_data = None
    return joint_subject_data

def runTracking(data_instance, save_dir, title_name, cache_data = False):
    pickle_dir = save_dir + "/pickles" + "/"
    image_dir = save_dir + "/images" + "/"
    mat_dir = save_dir + "/mats" + "/"
    voicecoil_dir = save_dir + "/voicecoil" + "/"
    save_suffix = ''
    os.makedirs(pickle_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mat_dir, exist_ok=True)
    os.makedirs(voicecoil_dir, exist_ok=True)
    pickle_name = pickle_dir + title_name + '.p'

    tared_frames = tareAndRemoveEdges(data_instance["frames"], buffer)


    if os.path.isfile(pickle_name) and cache_data:
        print('loading')
        cur_touch_trajectory_set = pickle.load(open(pickle_name, "rb"))
    else:
        print('not loading')
        print("Starting multi tracking")
        cur_touch_trajectory_set = cur_touch_trajectory_set = generateOptimalTrajectories(tared_frames)
        pickle.dump(cur_touch_trajectory_set, open(pickle_name, "wb"))
    print("Tracking or loading done, making plots")

    makeTrajectoryPlots(cur_touch_trajectory_set, data_instance, title_name, image_dir, save_suffix, ignore_circles=False)
    tactors = cur_touch_trajectory_set[2]
    generateVCOuput(tactors, voicecoil_dir, data_instance['emotion'], cutoff=1.0/8.0, copy_mainpath=False)
    plotSignalIntensities(cur_touch_trajectory_set, image_dir,title_name)


def main(argv):
    data_file = "data/subject_data_0.p"
    dataset = loadData(data_file)

    save_dir = "tracking_results/"
    runTracking(dataset[0], save_dir, "example_instance", cache_data = True)

if __name__ == "__main__":
    main(sys.argv[1:])


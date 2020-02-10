import csv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import math
import os
import scipy.signal
import sys
import getopt
import pickle
import peakutils
from constants import *
import imageio

def plotHeatmaps(frames, out_name, plot_range = None):
    os.makedirs(out_name, exist_ok=True)
    if plot_range is None:
        plot_range = len(frames)
    out_drawing = frames[:,:,:]
    max_val = np.amax(frames[:,:,:])
    out_drawing[np.where(frames == -10000)] = 0;
    max_val = np.nanmax(frames)
    for i in range(plot_range):
        plt.imshow(out_drawing[i], vmax=max_val, vmin = 0)
        plt.colorbar()
        plt.savefig(out_name +'/' + str(i).zfill(4) + '.png', bbox_inches='tight')
        plt.close()

def plotGif(frames, out_folder, out_name, title = None, highlights = None, polygons = [], plot_range = None, fps = 10, frequency=1):
    os.makedirs(out_folder, exist_ok=True)
    if plot_range is None:
        plot_range = len(frames)
    out_drawing = np.copy(frames)
    max_val = np.amax(frames[:,:,:])
    out_drawing[np.where(frames == -10000)] = 0;
    max_val = np.nanmax(frames)
    images = []
    filenames = []
    for i in range(0,plot_range,frequency):
        fig, ax = plt.subplots();
        plt.axis('off')
        im = ax.imshow(out_drawing[i], vmax=max_val, vmin = 0)
        if highlights is not None: 
            for hl in highlights[i]:
                circle = plt.Circle((hl['col'], hl['row']), 1, color=hl['color'])
                #ax.text(hl['col'],hl['row'],'.', ha="center", va="center", color=hl['color'], fontsize=64)
                ax.add_artist(circle)
        for poly in polygons:
            cur_poly = plt.Polygon(poly, fill=False, edgecolor='r')
            ax.add_patch(cur_poly)
        cbar = fig.colorbar(im)
        cbar.set_label("Pressure (psi)")
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        images.append(image)
        plt.close()
    imageio.mimsave(out_folder + '/' + out_name +'.gif', images, fps=10)

def plotGestureHistos(touches_data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gesture_counts = {}
    for touch in touches_data:
        if touch["emotion"] not in gesture_counts.keys():
            gesture_counts[touch["emotion"]] = {}
        if touch["gesture"] not in gesture_counts[touch["emotion"]].keys():
            gesture_counts[touch["emotion"]][touch["gesture"]] = 0
        gesture_counts[touch["emotion"]][touch["gesture"]] += 1
    for emotion in gesture_counts.keys():
        cur_dict = gesture_counts[emotion]
        plt.bar(range(len(cur_dict)), cur_dict.values(), align='center')
        plt.xticks(range(len(cur_dict)), cur_dict.keys())
        plt.title(emotion)
        plt.savefig(out_dir +'/' + emotion + '.png')
        plt.close()

def multiPlotHeatmaps(frames_array, out_name):
    num_plots = len(frames_array)
    os.makedirs(out_name, exist_ok=True)
    plt.figure(1)
    for i in range(len(frames_array[0])):
        for j in range(num_plots):
            plt.subplot(num_plots, 1, j+1)
            frames = frames_array[j]
            out_drawing = frames[:,:,:]
            max_val = np.amax(frames[:,:,:])
            out_drawing[np.where(frames == -10000)] = 0;
            max_val = np.nanmax(frames)
            plt.imshow(out_drawing[i], vmax=max_val)
            plt.colorbar()
        plt.savefig(out_name +'/' + str(i).zfill(4) + '.png', bbox_inches='tight')
        plt.close()

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import *
from scipy.signal import butter, lfilter, freqz
from constants import filter_val

predent = -.9
base_freq = 20
out_freq = 500
transition_time = .5
transition_time_end = .1
delay_buffer = 1.5
delay_buffer_end = 3

def generateVCOuput(tactors, out_location, out_name, cutoff = 1, copy_mainpath = False):
	max_length = 0
	for tactor in tactors:
		if len(list(tactor.actuator_locs.keys())) > 0:
			current_max_length = max(list(tactor.actuator_locs.keys()))
			if current_max_length > max_length:
				max_length = current_max_length

	if copy_mainpath:
		for tactor in tactors:
			for i in range(max_length + 1):
				if i not in list(tactor.main_path_locs.keys()):
					tactor.main_path_locs[i] = [0, 0, 0]

		for tactor in tactors:
			tactor.velocity_scaled_locs = [loc[2] for loc in tactor.main_path_locs.values()]

	else:
		for tactor in tactors:
			for i in range(max_length + 1):
				if i not in list(tactor.actuator_locs.keys()):
					tactor.actuator_locs[i] = [0, 0, 0]

		for tactor in tactors:
			tactor.velocity_scaled_locs = [loc[2] for loc in tactor.actuator_locs.values()]


	this_base_freq = base_freq

	vc_patterns = {}
	for i in range(8):
		vc_patterns[i] = []

	vc_patterns[0].append({'base_freq': this_base_freq, 'out_freq': out_freq, 'predent': predent,
						   'pressures': tactors[0].velocity_scaled_locs, 'duration': max_length + 1, 'start': 0})
	vc_patterns[1].append({'base_freq': this_base_freq, 'out_freq': out_freq, 'predent': predent,
						   'pressures': tactors[1].velocity_scaled_locs, 'duration': max_length + 1, 'start': 0})
	vc_patterns[2].append({'base_freq': this_base_freq, 'out_freq': out_freq, 'predent': predent,
						   'pressures': tactors[2].velocity_scaled_locs, 'duration': max_length + 1, 'start': 0})
	vc_patterns[3].append({'base_freq': this_base_freq, 'out_freq': out_freq, 'predent': predent,
						   'pressures': tactors[3].velocity_scaled_locs, 'duration': max_length + 1, 'start': 0})
	vc_patterns[4].append({'base_freq': this_base_freq, 'out_freq': out_freq, 'predent': predent,
						   'pressures': tactors[4].velocity_scaled_locs, 'duration': max_length + 1, 'start': 0})
	vc_patterns[5].append({'base_freq': this_base_freq, 'out_freq': out_freq, 'predent': predent,
						   'pressures': tactors[5].velocity_scaled_locs, 'duration': max_length + 1, 'start': 0})
	vc_patterns[6].append({'base_freq': this_base_freq, 'out_freq': out_freq, 'predent': predent,
						   'pressures': tactors[6].velocity_scaled_locs, 'duration': max_length + 1, 'start': 0})
	vc_patterns[7].append({'base_freq': this_base_freq, 'out_freq': out_freq, 'predent': predent,
						   'pressures': tactors[7].velocity_scaled_locs, 'duration': max_length + 1, 'start': 0})

	interleaved_out = patterns_to_VC(vc_patterns, predent, transition_time, out_freq, delay_buffer, cutoff=cutoff,
									 filter=filter_val)

	for i in range(8):
		plt.plot([t / 500 for t in range(len(interleaved_out[i::8]))], interleaved_out[i::8])
	plt.xlabel('time (s)')
	plt.ylabel('voicecoil current (A)')
	plt.show()

	f = open(out_location + "/" + out_name + ".txt", "w")

	f.write(" ".join([str(elem) for elem in interleaved_out]))

def stroke_dent(freq=0,  predent=0, duration=0, intensity=0, start=0):
    predent_offset = np.pi/2+np.arcsin(predent)

    predent_x = np.linspace(start = 3*np.pi/2 - predent_offset, stop = 3*np.pi/2 + predent_offset, num=freq*duration*(2*predent_offset)/(2*np.pi - 2*predent_offset))

    postdent_x = np.linspace(start=3*np.pi/2 + predent_offset, stop =2*np.pi + 3*np.pi/2 - predent_offset, num=freq*duration, endpoint=True )

    signal = np.array([])

    signal = np.append(signal, (np.sin(predent_x)))

    signal = np.append(signal, intensity*(-predent + np.sin(postdent_x)) + predent)

    return signal

def normalizeVCs(vc_patterns, predent, sine_freq=0, sine_intensity = 0):
	#min_pressure = float('inf')
	#max_pressure = -float('inf')
	min_pressure = min([min(vc_pattern['pressures']) for index in range(8) for vc_pattern in vc_patterns[index]])
	for index in range(8):
		for vc_pattern in vc_patterns[index]:
			vc_pattern['pressures'] -= min_pressure

	max_pressure = max([max(vc_pattern['pressures']) for index in range(8) for vc_pattern in vc_patterns[index]])
	for index in range(8):
		#if not normalize_to_all:
		#	if len(vc_patterns[index]) > 0:
		#		max_pressure = max([max(vc_pattern['pressures']) for vc_pattern in vc_patterns[index] if
		#								  not 'exclude_normalization' in list(vc_pattern.keys())])
		for vc_pattern in vc_patterns[index]:
			local_max_pressure = max([max(vc_pattern['pressures']) for vc_pattern in vc_patterns[index]])
			if local_max_pressure + min_pressure > 0.00001:
				vc_pattern['pressures'] /= max_pressure
				vc_pattern['pressures'] *= 1
				vc_pattern['pressures'] *= (-predent + max(vc_pattern['pressures']))/max(vc_pattern['pressures'])
				vc_pattern['pressures'] += predent
			else:
				vc_pattern['pressures'] += predent


def scaleVCs(vc_patterns, cutoff=1):
	end_time = 0
	min_start = float('inf')
	overall_min = float('inf')
	for index in range(8):
		for vc_pattern in vc_patterns[index]:
			current_min = min(vc_pattern['pressures'])
			if current_min < overall_min:
				overall_min = current_min

	# We need to prevent jumps, so add some zeros at start and end
	for index in range(8):
		for vc_pattern in vc_patterns[index]:
			zero_length = int(vc_pattern['base_freq']/5)
			vc_pattern['start'] = vc_pattern['start'] - 1*zero_length
			vc_pattern['duration'] = vc_pattern['duration'] + 2*zero_length
			#vc_pattern['pressures'] =  np.append([overall_min]*(zero_length), vc_pattern['pressures'])
			#vc_pattern['pressures'] =  np.append(vc_pattern['pressures'],[overall_min]*zero_length)

			new_xs = np.linspace(0, zero_length, zero_length+1, endpoint=True)
			f = interp1d([0, zero_length], [overall_min] + [vc_pattern['pressures'][0]], kind='linear')
			pre_addendum = f(new_xs)
			vc_pattern['pressures'] = np.append(pre_addendum[0:-1], vc_pattern['pressures'])


			new_xs = np.linspace(0, zero_length, zero_length+1, endpoint=True)
			f = interp1d([0, zero_length], [vc_pattern['pressures'][-1]] + [overall_min], kind='linear')
			post_addendum = f(new_xs)
			vc_pattern['pressures'] = np.append(vc_pattern['pressures'],post_addendum[1:])


	for index in range(8):
		for vc_pattern in vc_patterns[index]:
			
			if vc_pattern['start'] < min_start:
				min_start = vc_pattern['start']
	for index in range(8):
		for vc_pattern in vc_patterns[index]:
			vc_pattern['start'] = int(vc_pattern['start'] - min_start)
			upscale_factor = vc_pattern['out_freq']/vc_pattern['base_freq']
			vc_pattern['start'] *= upscale_factor
			vc_pattern['start'] = int(vc_pattern['start'])
			vc_pattern['duration'] *= upscale_factor
			vc_pattern['duration'] = int(vc_pattern['duration'])
			pressure_list = vc_pattern['pressures']
			new_xs = np.linspace(0, len(pressure_list)-1, num=int(len(pressure_list)*upscale_factor), endpoint=True)
			f = interp1d(range(len(pressure_list)), pressure_list,'linear')
			vc_pattern['pressures'] = f(new_xs)
			vc_pattern['pressures'] = np.array([min(p, 1*max(vc_pattern['pressures'])) for p in vc_pattern['pressures']])
			if vc_pattern['start'] + vc_pattern['duration'] > end_time:
				end_time = vc_pattern['start'] + vc_pattern['duration']
	return end_time

def filterVCs(vc_patterns, cutoff, fs):
	for index in range(8):
		for vc_pattern in vc_patterns[index]:
			vc_pattern['pressures'] = butter_lowpass_filter(vc_pattern['pressures'], cutoff, fs)

def cutoffTops(vc_patterns, cutoff, max_pressure = None):
	min_pressure = min([min(vc_pattern['pressures']) for index in range(8) for vc_pattern in vc_patterns[index]])
	max_pressure = max([max(vc_pattern['pressures']) for index in range(8) for vc_pattern in vc_patterns[index]])
	pressure_range = max_pressure - min_pressure
	for index in range(8):
		for vc_pattern in vc_patterns[index]:
			peak_pressures = []
			for i in range(1, len(vc_pattern['pressures']) - 1):
				if vc_pattern['pressures'][i] > vc_pattern['pressures'][i-1] and vc_pattern['pressures'][i] > vc_pattern['pressures'][i+1]:
					cur_pressure_range = max([vc_pattern['pressures'][i] - vc_pattern['pressures'][i-1], vc_pattern['pressures'][i] - vc_pattern['pressures'][i+1]])
					if (cur_pressure_range/pressure_range > cutoff):
						new_val = (vc_pattern['pressures'][i+1] + vc_pattern['pressures'][i-1])/2
						vc_pattern['pressures'][i] = new_val
						vc_pattern['pressures'][i+1] = new_val
						vc_pattern['pressures'][i-1] = new_val

def add_pattern(out_array, index, pattern):
	signal = pattern['pressures']
	out_array[index, int(pattern['start']):int((pattern['start'] + len(signal)))] = signal

def patterns_to_VC(vc_patterns, predent, transition_time, frequency, delay_buffer, cutoff=1, sine_freq = 0, sine_intensity=0, filter = 4):
	if filter is not None:
		filterVCs(vc_patterns, filter, 20)
	if cutoff != 1:
		cutoffTops(vc_patterns, cutoff)
	total_duration = scaleVCs(vc_patterns, cutoff=cutoff)
	normalizeVCs(vc_patterns, predent, sine_freq = sine_freq, sine_intensity = sine_intensity)

	out_array = np.ones((8, total_duration))*predent
	for index in range(8):
		for pattern in vc_patterns[index]:
			add_pattern(out_array, index, pattern)
	max_difference = 0
	for index in range(8):
		last_val = out_array[index, -1]
		difference = last_val - predent
		num_points = int(difference*100)
		if max_difference < num_points:
			max_difference = num_points
	zero_cat = np.zeros((8, num_points))
	out_array = np.concatenate((out_array, zero_cat), axis=1)
	for index in range(8):
		last_val = out_array[index, -num_points-1]
		extra_postcat = np.linspace(start = last_val, stop=predent, num=max_difference)
		if len(extra_postcat) > 0:
			out_array[index, -max_difference:] = extra_postcat
	precat = np.linspace(start=0, stop=predent, num=int(frequency*transition_time+1))
	precat = np.tile(precat, (8,1))
	buffercat = np.linspace(start = predent, stop=predent, num=int(frequency*delay_buffer+1))
	buffercat = np.tile(buffercat, (8,1))
	buffercat_end = np.linspace(start=predent, stop=predent, num=int(frequency*delay_buffer_end + 1))
	buffercat_end = np.tile(buffercat_end, (8, 1))
	postcat = np.linspace(start=predent, stop=0, num=int(frequency*transition_time_end+1))
	postcat = np.tile(postcat, (8,1))
	out_array = np.concatenate((precat, buffercat, out_array, buffercat_end, postcat), axis=1)
	interleaved_out = np.zeros((out_array.size,))
	for index in range(8):
		interleaved_out[index::8] = out_array[index, :]
	return interleaved_out

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
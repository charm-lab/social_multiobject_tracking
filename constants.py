gesture_names = ['Poke', 'Massage', 'Shake', 'Tap', 'Pat', 'Press', 'Squeeze', 'Stroke', 'Strassage']
gesture_nums = range(len(gesture_names))
emotion_names = ['Happiness', 'Calming', 'Love', 'Gratitude', 'Attention', 'Sadness']
emotion_map = {'1':'Happiness', '2':'Calming', '3':'Love', '4':'Gratitude', '5':'Attention', '6':'Sadness'}
emotion_nums = ['1', '2','3','4','5','6']
out_emotion_nums = ['5','4','1','2','3','6']

# Parameters for each touch parsed from decision trees
happiness_params = {"gesture": ['y', "Shake"], "total_time": ['g', 29.5], 'spread_ranges_std_intensity' : ['l', .205]}
calming_params = {"times_intensity_std" : ['l', 7.554], "spread_extrema_mean_intensity" : ['g', .758], "extrema_frequency_location" : ['g', .129],\
                    "gesture": ['n', "Shake"], "gesture" : ['n', 'Tap'], "total_time" : ['g', 46.5]}
love_params = {"times_intensity_std" : ['l', 7.554], "spread_extrema_mean_intensity" : ['l', .758], "spread_ranges_mean_intensity" : ['l', .143], \
            "extrema_frequency_location": ['l', .193], "total_time" : ['l', 128], "gesture": ['n' "Shake"], "gesture" : ['n', 'Tap'], "total_time" : ['g', 46.5]}
gratitude_params = {"total_time" : ['l', 46.5], "gesture": ['n', "Shake"]}
attention_params = {"gesture": ['y', 'Tap']}
sadness_params = {"gesture": ['n', "Shake"], "gesture" : ['n', 'Tap'], "total_time" : ['g', 46.5], "times_intensity_std" : ['g', 7.554]} 

params_list = [happiness_params, calming_params, love_params, gratitude_params, attention_params, sadness_params]
frame_scaling_factor = 7

#multi object tracking constants
source_sink_cost = 8
std_dev = 1.25
max_dist = 50
blurring = 3
filter_val = 4
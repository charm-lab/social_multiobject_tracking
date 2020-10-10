multiobject tracking algorithm

Works for python3 in Linux or Mac OS, Windows is not supported due to dependency on https://github.com/nwojke/mcf/

Download https://github.com/nwojke/mcf/, put into "dependencies" folder. Follow instructions provided there.

Applicable data can be found at:

https://stanford.box.com/v/sparse-social-touch
folder:pressure_data

This data includes a description of the data.

Data runs assuming a mapping to 8 1DoF actuators in a 2x4 array. By default it loads "data/subject_data_0.p", and runs the first example data. Modify this to load different data.

The core of the multi-object tracking algorithm is performed in muli_tracking.py. The workspace is defiend in base_poly_funcs.py.

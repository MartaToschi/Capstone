#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import sys
import h5py
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------

def load_features(file_name):
    # dictionary to load features into
    features = {}

    # load file using the HDF5 library
    f = h5py.File(file_name, 'r')

    print("Loading features from {0} ... ".format(file_name))

    # loop over each feature contained in the file
    for feature_name in f.keys():
        # convert to numpy format and store in dictionary
        x = np.array(f[feature_name])
        print(feature_name, "{0}x{1}".format(x.shape[0], x.shape[1]))
        features[feature_name] = x

    return features

#------------------------------------------------------------------------------

video_features = load_features('Hollywood-dev/features/Armageddon_visual.mat')
audio_features = load_features('Hollywood-dev/features/Armageddon_auditory.mat')

print (len(video_features))
print (len(audio_features))

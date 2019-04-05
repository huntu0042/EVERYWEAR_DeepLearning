import pickle
import os
import collections
import numpy as np
import scipy.io as sio
DATA_PATH = "/Users/khosungpil/Downloads/CP-VVITON(men_tshirts)/data/pose/"
TRAIN_LABEL_FILE = "/Users/khosungpil/Downloads/CP-VVITON(men_tshirts)/data/viton_train_images.txt"

def _load_and_process_posedata(label_file, datadict):
    image_pairs = open(label_file).read().splitlines()
    image_metadata = []
    for item in image_pairs:
        info_data = {}
        image_pair = item.split()
        pose_data = sio.loadmat(DATA_PATH + image_pair[0][:-4] + ".mat")
        subset = np.array(pose_data["subset"])
        candidate = np.array(pose_data["candidate"])
        info_data['subset'] = subset
        info_data['candidate'] = candidate
        datadict[image_pair[0][:-4]] = info_data
    
    return datadict

data = {}
data = _load_and_process_posedata(TRAIN_LABEL_FILE, data)
"""
print(data)
"""

with open('pose_tshirts.pkl', 'wb') as f:
    pickle.dump(data, f, protocol=2)

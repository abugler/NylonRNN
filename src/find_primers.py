import pretty_midi
import numpy as np
from encoding import encoding_to_LSTM
import os
"""
This file finds the "primer matrix" for the feed_forward of the neural network
"""

absolute_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\NylonRNN\\"
data_path = "data\\classical_guitar_npdata\\"
primer_path = "data\\primer.npy"
try:
    np_list = os.listdir(data_path)
except FileNotFoundError:
    np_list = os.listdir(absolute_path + data_path)
    primer_path = absolute_path + primer_path

primer_matrix = np.empty((50, len(np_list)))
for i in range(len(np_list)):
    midi_data = np.load(data_path + path)
    primer_matrix[:, i] = midi_data[:, 0]

np.save(primer_path, primer_matrix)

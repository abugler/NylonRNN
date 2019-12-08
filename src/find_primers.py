import pretty_midi
import numpy as np
from encoding import encoding_to_LSTM
"""
This file finds the "primer matrix" for the feed_forward of the neural network
"""

absolute_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\EtudeRNN\\"
data_path = "data\\Classical_Guitar_classicalguitarmidi.com_MIDIRip\\"
song_list_filepath = "data\\classical_guitar_training_set"
primer_path = "src\\primer.npy"
try:
    with open(song_list_filepath, 'r') as song_list_file:
        song_list = song_list_file.readlines()
except FileNotFoundError:
    data_path = absolute_path + data_path
    song_list_filepath = absolute_path + song_list_filepath
    primer_path = absolute_path + primer_path
    with open(song_list_filepath, 'r') as song_list_file:
        song_list = ''.join(song_list_file.read()).split('\n')

encoded_matrices = []
for path in song_list:
    midi_data = pretty_midi.PrettyMIDI(data_path + path)
    encoded_matrices += encoding_to_LSTM(midi_data)

primer_matrix = np.empty((50, len(encoded_matrices) * 24))
for i, mat in zip(range(len(encoded_matrices)), encoded_matrices):
    primer_matrix[:, i: i + 24] = mat[:, 0:24]

np.save(primer_path, primer_matrix)

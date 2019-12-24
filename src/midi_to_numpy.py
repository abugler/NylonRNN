from encoding import encoding_to_LSTM
import pretty_midi
import numpy as np


absolute_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\EtudeRNN\\"
data_path = "data\\Classical_Guitar_classicalguitarmidi.com_MIDIRip\\"
song_list_filepath = "data\\classical_guitar_training_set"
encode_path = "data\\classical_guitar_npdata\\"
try:
    with open(song_list_filepath, 'r') as song_list_file:
        song_list = ''.join(song_list_file.read()).split('\n')
except FileNotFoundError:
    data_path = absolute_path + data_path
    song_list_filepath = absolute_path + song_list_filepath
    encode_path = absolute_path + encode_path
    with open(song_list_filepath, 'r') as song_list_file:
        song_list = ''.join(song_list_file.read()).split('\n')

encoded_matrices = []
iter = 0
for path in song_list:
    if path.find('val') == -1:
        break
    midi_data = pretty_midi.PrettyMIDI(data_path + path)
    for matrix in encoding_to_LSTM(midi_data):
        np.save(encode_path + "val" + str(iter) + ".npy", matrix)
        iter += 1
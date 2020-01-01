from encoding import encoding_to_LSTM
import pretty_midi
import numpy as np

absolute_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\NylonRNN\\"
data_path = "data\\Classical_Guitar_classicalguitarmidi.com_MIDIRip\\"
song_list_filepath = "data\\classical_guitar_training_set"
encode_path = "data\\classical_guitar_npdata\\"
timestep_path = "data\\timestep.npy"
try:
    with open(song_list_filepath, 'r') as song_list_file:
        song_list = ''.join(song_list_file.read()).split('\n')
except FileNotFoundError:
    data_path = absolute_path + data_path
    song_list_filepath = absolute_path + song_list_filepath
    encode_path = absolute_path + encode_path
    timestep_path = absolute_path + timestep_path
    with open(song_list_filepath, 'r') as song_list_file:
        song_list = ''.join(song_list_file.read()).split('\n')

iter = 0
starting_timesteps = []
for path in song_list:
    midi_data = pretty_midi.PrettyMIDI(data_path + path)
    if iter == 46:
        print()
    for matrix, timestep in encoding_to_LSTM(midi_data):
        np.save(encode_path + str(iter) + ".npy", matrix)
        starting_timesteps.append(timestep)
        iter += 1

np.save(timestep_path, np.array(starting_timesteps))
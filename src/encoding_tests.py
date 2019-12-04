from src import train_rnn
import numpy as np
import datetime
import pretty_midi

training_list = "\\data\\classical_guitar_training_set"
training_folder = "\\data\\Classical_Guitar_classicalguitarmidi.com_MIDIRip\\"
root = ""
full_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\MelodyLSTM"

# This is done to keep Pycharm from giving me a hard time.
training_files = None
while training_files is None:
    try:
        with open(root + training_list) as file:
            training_files = np.array(file.read().split("\n"))
    except FileNotFoundError:
        root = full_path

def test_generate():
    ## Sample test
    ## This is simply done to check if the functions crashes or not.
    np.random.seed(int(datetime.datetime.now().microsecond))

    sample_file = training_files[np.random.randint(0, training_files.shape[0])]
    # sample_file = "Giuliani_Papillon_Op50_No11.mid"
    midi_data = pretty_midi.PrettyMIDI(root + training_folder + sample_file)

    print(train_rnn.encoding_to_LSTM(midi_data))

def test_generate_all():
    passed_all = True
    for training_file in training_files:
        midi_data = pretty_midi.PrettyMIDI(root + training_folder + training_file)
        try:
            train_rnn.encoding_to_LSTM(midi_data)
        except:
            print(training_file)
            passed_all = False
    assert passed_all

# test_generate_all()

def test_generate_problematic():
    """
    This test tests the pieces that was causing the "test_generate_all" test to fail
    """
    problematic_files = [
        "Bach_Partita_No1_BWV825_Gigue.mid",
        "Falu_Mishi.mid"
    ]
    passed_all = True
    for training_file in problematic_files:
        midi_data = pretty_midi.PrettyMIDI(root + training_folder + training_file)
        try:
            train_rnn.encoding_to_LSTM(midi_data)
        except:
            print(training_file)
            passed_all = False
    assert passed_all

test_generate_problematic()
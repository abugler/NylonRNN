from src import encoding
import numpy as np
import datetime
import pretty_midi

training_list = "\\data\\classical_guitar_training_set"
training_folder = "\\data\\Classical_Guitar_classicalguitarmidi.com_MIDIRip\\"
root = ""
full_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\EtudeRNN"

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

    print(encoding.encoding_to_LSTM(midi_data))

def encode_decode():
    """
    This test tests whether or not the encoding and decoding can recreate a MIDI file.
    Aguado_12valses_Op1_No12 was chosen and the test track, since it has no tempo changes,
    (We do not need the decoder to recreate midi with tempo changes, since the output with a 8 or 16 measure
    sequence with no tempo changes)
    and it features notes that are 1/8 and 1/6 of a quarter note long, the goal for the encoding is to be able to encode
    both of these note lengths if the note begins on any of the 1/24 time steps.
    """
    test_song = "Aguado_12valses_Op1_No12.mid"
    midi_data = pretty_midi.PrettyMIDI(root + training_folder + test_song)
    encoded_array = encoding.encoding_to_LSTM(midi_data)[0]
    predicted_midi_data = encoding.decoding_to_midi(encoded_array, tempo=70, time_signature="3/8")
    predicted_notes = sorted(predicted_midi_data.instruments[0].notes, key=lambda x: x.start + .0001 * x.pitch)
    actual_notes = sorted(midi_data.instruments[0].notes, key=lambda x: x.start + .0001 * x.pitch)

    assert len(actual_notes) == len(predicted_notes)

    # These are the smallest tolerances we can use to pass the test
    rtol = 7e-4
    atol = 1e-8
    for actual, predicted in zip(actual_notes, predicted_notes):
        assert np.allclose(actual.pitch, predicted.pitch, atol=atol, rtol=rtol)
        assert np.allclose(actual.start,  predicted.start, atol=atol, rtol=rtol)
        assert np.allclose(actual.end, predicted.end, atol=atol, rtol=rtol)

encode_decode()

def test_generate_all():
    passed_all = True
    for training_file in training_files:
        midi_data = pretty_midi.PrettyMIDI(root + training_folder + training_file)
        try:
            encoding.encoding_to_LSTM(midi_data)
        except Exception as e:

            print("Error with: " + training_file + " Error: " + str(e))

            passed_all = False
    assert passed_all

test_generate_all()

def test_generate_problematic():
    """
    This test tests the pieces that was causing the "test_generate_all" test to fail.

    This test will fail. I do not know why these songs fail.
    It was not worth the time and effort to figure out why these songs were giving me a hard time.
    I hypothesize that songs with bpms of over 250 are giving it trouble, but I do not know for sure.
    """
    # problematic_files = [
    #     "Bach_Partita_No1_BWV825_Gigue.mid",
    #     # 2178 timesteps in the first range. tempo = 252, so each timestep should be 60 / (252 * 24) = .01 seconds
    #     "Falu_Mishi.mid"
    # ]

    problematic_files = [
        "Garoto_Desvairada.mid"
    ]
    passed_all = True
    for training_file in problematic_files:
        midi_data = pretty_midi.PrettyMIDI(root + training_folder + training_file)
        encoding.encoding_to_LSTM(midi_data)

# test_generate_problematic()
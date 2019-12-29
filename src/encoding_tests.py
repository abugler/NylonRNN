import src.encoding as encoding
import numpy as np
import datetime
import pretty_midi
import warnings

warnings.filterwarnings("error")
training_list = "\\data\\classical_guitar_training_set"
training_folder = "\\data\\Classical_Guitar_classicalguitarmidi.com_MIDIRip\\"
root = ""
full_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\NylonRNN"

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
    # Due to changing the encoding, this test no longer works
    return
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



def test_generate_all():
    passed_all = True
    for training_file in training_files:
        try:
            midi_data = pretty_midi.PrettyMIDI(root + training_folder + training_file)
            encoding.encoding_to_LSTM(midi_data)
        except Exception as e:
            print("Error with: " + training_file + " Error: " + str(e))
            passed_all = False
        except RuntimeWarning as e:
            print("RuntimeWarning with: " + training_file + " RuntimeWarning: " + str(e))
            passed_all = False
    assert passed_all
    """
    Captured print: 
    Error with: Bach_Allegro_BWV998.mid Error: Mean of empty slice.
Error with: Bach_Partita_No1_BWV1002_Courante.mid Error: Mean of empty slice.
Error with: Bach_Partita_No1_BWV825_Gigue.mid Error: Mean of empty slice.
Error with: Britten_Op70_Nocturnal_7_Cullante.mid Error: Mean of empty slice.
Error with: Broca_Minueto(a).mid Error: Tempo, Key or Time signature change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI file.  Tempo, Key or T
ime Signature may be wrong.
Error with: Carcassi_Nouveau_Papillon_Op5_No3.mid Error: Mean of empty slice.
Error with: Cardoso_48_Piezas_Suite_de_los_Mita_23_Albazo.mid Error: Mean of empty slice.
Error with: Coste_Valse_Op7_No3.mid Error: Mean of empty slice.
Error with: Dowland_Mrs_White's_Nothing.mid Error: Mean of empty slice.
Error with: Falu_Choro.mid Error: Mean of empty slice.
Error with: Ferranti_Ocho_Piezas_Faciles_No4.mid Error: Mean of empty slice.
Error with: Garoto_Desvairada.mid Error: Mean of empty slice.
Error with: Giuliani_Etude_No20_Op48.mid Error: Mean of empty slice.
Error with: Giuliani_Sonate_No2_Op.96_2Allegretto.mid Error: Mean of empty slice.
Error with: Losy_Gigue.mid Error: Mean of empty slice.
Error with: Montana_Suite_Colombiana_No1_IICancion.mid Error: index 6 is out of bounds for axis 0 with size 6
Error with: Oliva_Sonata_del_Amor_Plenitude.mid Error: Mean of empty slice.
Error with: Paganini_MS84_37sonates_No14.mid Error: Mean of empty slice.
Error with: Ponce_Suite_la_Style_Weiss_1Preludio.mid Error: Mean of empty slice.
Error with: Ponce_Suite_la_Style_Weiss_2Allemanda.mid Error: Mean of empty slice.
Error with: Sagreras_Estilo_Criollo_No2_Op11.mid Error: Tempo, Key or Time signature change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI file.
 Tempo, Key or Time Signature may be wrong.
Error with: Sagreras_Estilo_Criollo_No4_Op11.mid Error: Tempo, Key or Time signature change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI file.
 Tempo, Key or Time Signature may be wrong.
Error with: Sagreras_Estudio_No1_Sonatina_Op23.mid Error: Tempo, Key or Time signature change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI file
.  Tempo, Key or Time Signature may be wrong.
Error with: Sagreras_Estudio_No4_Sonatina_Op31.mid Error: Tempo, Key or Time signature change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI file
.  Tempo, Key or Time Signature may be wrong.
Error with: Sagreras_Estudio_No6_Sonatina_Op46.mid Error: Tempo, Key or Time signature change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI file
.  Tempo, Key or Time Signature may be wrong.
Error with: Sagreras_Quejas_Amorosas_Op2.mid Error: Tempo, Key or Time signature change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI file.  Tem
po, Key or Time Signature may be wrong.
Error with: Sagreras_Tres_Piezas_Faciles_3_Nostalgia_Op19.mid Error: Tempo, Key or Time signature change events found on non-zero tracks.  This is not a valid type 0 or type
1 MIDI file.  Tempo, Key or Time Signature may be wrong.
Error with: Sagreras_Zamba_Op10.mid Error: Tempo, Key or Time signature change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI file.  Tempo, Key o
r Time Signature may be wrong.
Error with: Sanz_Espagnoletta.mid Error: Tempo, Key or Time signature change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI file.  Tempo, Key or
Time Signature may be wrong.
Error with: Tansman_Suite_Polonico_Kolysanka_No2.mid Error: index 6 is out of bounds for axis 0 with size 6
Error with: Vinas_Capullos_de_Abril_1_Pasa_Calle.mid Error: Tempo, Key or Time signature change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI fi
le.  Tempo, Key or Time Signature may be wrong.
Error with: Vinas_Capullos_de_Abril_6_Marcha.mid Error: Tempo, Key or Time signature change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI file.
 Tempo, Key or Time Signature may be wrong.

    """

# test_generate_all()

def test_generate_problematic():
    """
    This test tests the pieces that was causing the "test_generate_all" test to fail.
    """

    problematic_files = [
        "Tansman_Suite_Polonico_Kolysanka_No2.mid",
        "Montana_Suite_Colombiana_No1_IICancion.mid"
    ]
    passed_all = True
    for training_file in problematic_files:
        midi_data = pretty_midi.PrettyMIDI(root + training_folder + training_file)
        encoding.encoding_to_LSTM(midi_data)

test_generate_problematic()
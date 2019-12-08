import pretty_midi
import numpy as np

# E2 is the lowest note on a Standard classical Guitar
E2 = 40
# B5 is the highest fretted note on a standard classical guitar
B5 = 83

def encoding_to_LSTM(midi_data: pretty_midi.PrettyMIDI):
    """
    The encoding for this data is specific to solo classical guitar pieces with no pinch harmonics nor percussive elements.

    The encoding for LSTM will be a np.ndarray of 50 rows and d columns, where there are d time steps.
    The first 44 rows will be for marking whether or not the corresponding pitch will be played. Row 0 will correspond with E2,
    the lowest note on classical guitar, row 1 will correspond with F2, row 2 will correspond with F#2, and so on,
    until row 43, which corresponds with B5, and is the highest non-harmonic note on classical guitar.

    The last 6 rows correspond with whether or not a specific note if plucked or held from the previous timestep.
    For example, if a 1 exists in row 44, then the lowest note found in above 44 rows is to be plucked in this timestep.
    If it is 0, then the lowest note found in the above 44 rows is held from the previous timestep. This is the same for rows
    45-49, where each row corresponds with the 2nd-6th lowest note, respectively. The rationale for this part of the encoding is
    to differentiate between many of the same note being played at the same time and a not being held.

    Each timestep is 1/24 of a beat.  This is to account for both notes that last 1/8 of a beat, and notes that last 1/3 of a beat.
    As most songs' shortest notes are roughly either 1/6 or 1/8 of a beat, this will account for both.

    Midi_data will be segmented by tempo. Sections less than 8 beats of constant tempo will be ignored.

    :param midi_data: A pretty_midi.PrettyMidi object to be encoded
    :param tempo: Tempo of pretty_midi object.  Default is 100
    :return: encoded_matrices: A list of encoded matrices
    """
    beats_min = 8
    tempo_change_times, tempi = midi_data.get_tempo_changes()
    if tempo_change_times is None:
        tempo = midi_data.estimate_tempo()
        range_vectors = [np.arange(0, midi_data.get_end_time(), 1 / (24 * tempo))]
        range_tempi = [tempo]
    else:
        range_vectors = []
        range_tempi = []
        for i in range(len(tempi)):
            start_time = tempo_change_times[i]
            end_time = tempo_change_times[i + 1] if i < len(tempi) - 1 else midi_data.get_end_time()
            vector = np.arange(start_time, end_time, 1 / (tempi[i] / 60 * 24))
            if vector.shape[0] > beats_min * 24:
                range_vectors.append(vector)
                range_tempi.append(tempi[i])

    # This will only work with midi data with a single instrument
    def find_pluck_matrix(midi_data: pretty_midi.PrettyMIDI, vector: np.ndarray, piano_roll: np.ndarray, tempo: int):
        pluck_matrix = np.zeros((6, vector.shape[0]))
        section_notes = lambda _note:_note.start >= vector[0] and _note.end <= vector[-1] + tempo / 60 / 24
        notes = sorted(filter(section_notes, midi_data.instruments[0].notes), key=lambda x: x.pitch)
        section_start = vector[0]
        for note in notes:
            timestep = int(round((note.start - section_start) / 60 * tempo * 24, 0))

            simultaneous_notes = np.sum(piano_roll[0:(note.pitch - E2), timestep])
            pluck_matrix[simultaneous_notes, timestep] = 1
        return pluck_matrix

    encoded_matrices = []
    instrument = midi_data.instruments[0]

    def adjust_end_times(instrument, inc):
        for note in instrument.notes:
            note.end -= inc
            note.start += inc
        return instrument
    for vector, tempo in zip(range_vectors, range_tempi):
        midi_matrix = instrument.get_piano_roll(times=vector)[E2:B5 + 1, :]

        # Right now, midi_matrix is a matrix of velocities.
        # Let's change this so midi matrix is a matrix of whether the note is played or not
        one_hot = np.vectorize(lambda x: np.int(x != 0))
        midi_matrix = one_hot(midi_matrix)

        # Sometimes, start and end times overlap, and cause more than 6 notes to be detected, this fixes that
        inc = 0
        timestep = vector[1] - vector[0]
        max_simul = np.max(midi_matrix.sum(axis=0))
        max_timestep = np.argmax(midi_matrix.sum(axis=0))
        while max_simul > 6:
            if inc > 10:
                raise ValueError("There are more than 6 notes being played at once somewhere in this song, remove it."
                                 "Problematic Beats: " + str(max_timestep / 24))

            instrument = adjust_end_times(instrument, timestep / 10)
            midi_matrix = instrument.get_piano_roll(times=vector)[E2:B5 + 1, :]
            one_hot = np.vectorize(lambda x: np.int(x != 0))
            midi_matrix = one_hot(midi_matrix)
            max_simul = np.max(midi_matrix.sum(axis=0))
            max_timestep = np.argmax(midi_matrix.sum(axis=0))
            inc += 1

        pluck_matrix = find_pluck_matrix(midi_data, vector, midi_matrix, tempo)
        encoded_matrices.append(
            np.append(midi_matrix, pluck_matrix, axis=0)
        )

    return encoded_matrices

def decoding_to_midi(encoded_matrix, tempo=100, time_signature="4/4"):
    """
    Decodes a matrix encoded for LSTM back to a PrettyMIDI file
    :param encoded_matrix:
    :param tempo:
    :param time_signature:
    :return midi_data:
    """

    # Split encoded_matrix into piano_roll and pluck_matrix
    piano_roll = encoded_matrix[:B5 - E2 + 1, :]
    pluck_matrix = encoded_matrix[-6:, :]
    plucks_per_timestep = pluck_matrix.sum(axis=0)
    pluck_nonzero = plucks_per_timestep.nonzero()[0]

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_signature = time_signature.split('/')
    midi_data.time_signature_changes.append(
        pretty_midi.TimeSignature(int(time_signature[0]), int(time_signature[1]), 0)
    )
    midi_data.instruments.append(pretty_midi.Instrument(24, name="Guitar"))

    timesteps_per_second = tempo / 60 * 24

    for timestep in reversed(pluck_nonzero):
        plucks = np.array(pluck_matrix[:, timestep].nonzero(), dtype=np.int32).flatten()
        notes_played = np.array(piano_roll[:, timestep].nonzero(), dtype=np.int32).flatten()
        pitches = notes_played[plucks]
        for pitch in pitches:
            notes_equal_pitch = list(filter(lambda note: note.pitch == pitch + E2,
                                        midi_data.instruments[0].notes))
            if notes_equal_pitch:
                next_pluck = int(min(notes_equal_pitch,
                        key=lambda note: note.start).start \
                         * timesteps_per_second)
            else:
                next_pluck = 1e10
            note_timesteps = 1
            while timestep + note_timesteps < piano_roll.shape[1] \
                  and timestep + note_timesteps < next_pluck \
                    and piano_roll[pitch, timestep + note_timesteps]:
                note_timesteps += 1
            note_length = note_timesteps / timesteps_per_second
            begin = timestep / timesteps_per_second
            midi_data.instruments[0].notes.append(pretty_midi.Note(127, pitch + E2, begin, begin + note_length))

    return midi_data

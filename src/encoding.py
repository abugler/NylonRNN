import pretty_midi
import numpy as np

# E2 is the lowest note on a Standard classical Guitar
E2 = 40
# B5 is the highest fretted note on a standard classical guitar
B5 = 83

# A beat is 24 timesteps
beat_length = 24

# Generates the beat matrix template
beat_signals = [3, 4, 6, 8, 12, 24, 36, 48, 72, 96]
beat_template = np.zeros((len(beat_signals), max(beat_signals)))
for i in range(len(beat_signals)):
    curr = 0
    while curr < beat_template.shape[1]:
        beat_template[i, curr] = 1
        curr += beat_signals[i]

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

    Midi_data will be segmented by tempo and Time Signature. Sections less than 4 beats of constant tempo will be ignored.
    In addition to this, the number of timesteps into the song will be measured, so the beat matrix can be accurately aligned with it

    :param midi_data: A pretty_midi.PrettyMidi object to be encoded
    :param tempo: Tempo of pretty_midi object.  Default is 100
    :return: encoded_matrices: A list of encoded matrices
    """
    # First we want to find the locations in the midi track where the time signature changes
    beats_min = 4
    tempo_change_times, tempi = midi_data.get_tempo_changes()
    time_signature_changes = sorted(midi_data.time_signature_changes, key=lambda sign: sign.time)
    changes = []
    last_tempo = midi_data.estimate_tempo()
    i = 0
    j = 0
    while i + j < len(time_signature_changes) + tempo_change_times.shape[0]:
        if j == len(time_signature_changes) or \
                (i != tempo_change_times.shape[0] and
                 time_signature_changes[j].time >= tempo_change_times[i]):
            next_change = (tempo_change_times[i], tempi[i])
            last_tempo = tempi[i]
            i += 1
        else:
            next_change = (time_signature_changes[j].time, last_tempo)
            j += 1
        changes.append(next_change)

    # After we do so, we need to find the range vectors to pass into the function "PrettyMIDI.get_piano_roll" for the given
    # range and tempo
    if changes is None:
        # In my ancedotal observations, this function is not the most accurate, so we only use it if tempo data is not provided.
        tempo = midi_data.estimate_tempo()
        range_vectors = [np.arange(0, midi_data.get_end_time(), 1 / (beat_length * tempo))]
        range_tempi = [tempo]
    else:
        range_vectors = []
        range_tempi = []
        for i in range(len(changes)):
            start_time = changes[i][0]
            end_time = changes[i + 1][0] if i < len(changes) - 1 else midi_data.get_end_time()
            vector = np.arange(start_time, end_time, 1 / (changes[i][1] / 60 * beat_length))[:-1]
            if vector.shape[0] > beats_min * beat_length:
                range_vectors.append(vector)
                range_tempi.append(changes[i][1])

    # This will only work with midi data with a single instrument
    def find_attack_matrix(midi_data: pretty_midi.PrettyMIDI, vector: np.ndarray, piano_roll: np.ndarray, tempo: int):
        attack_matrix = np.zeros((6, vector.shape[0]))
        section_notes = lambda _note:_note.start >= vector[0] and _note.start <= vector[-1]
        notes = sorted(filter(section_notes, midi_data.instruments[0].notes), key=lambda x: x.pitch)
        section_start = vector[0]
        for note in notes:
            timestep = int(round((note.start - section_start) / 60 * tempo * beat_length, 0))

            simultaneous_notes = np.sum(piano_roll[0:(note.pitch - E2), timestep])
            attack_matrix[simultaneous_notes, timestep] = 1
        return attack_matrix

    encoded_matrices = []
    instrument = midi_data.instruments[0]

    def adjust_end_times(instrument, inc):
        for note in instrument.notes:
            note.end -= inc
            note.start += inc
        return instrument

    def overlap_check(instrument, midi_matrix):
        # Sometimes, start and end times overlap, and cause more than 6 notes to be detected, this fixes that
        inc = 0
        timestep = vector[1] - vector[0]
        max_simul = np.max(midi_matrix.sum(axis=0))
        while max_simul > 6:
            if inc > 10:
                return None
            instrument = adjust_end_times(instrument, timestep / 10)
            midi_matrix = instrument.get_piano_roll(times=vector)[E2:B5 + 1, :]
            one_hot = np.vectorize(lambda x: np.int(x != 0))
            midi_matrix = one_hot(midi_matrix)
            max_simul = np.max(midi_matrix.sum(axis=0))
            inc += 1
        return midi_matrix

    timesteps_passed = 0
    for vector, tempo in zip(range_vectors, range_tempi):
        # Right now, midi_matrix is a matrix of velocities.
        # Let's change this so midi matrix is a matrix of whether the note is played or not
        one_hot = np.vectorize(lambda x: np.int(x != 0))
        midi_matrix = one_hot(instrument.get_piano_roll(times=vector)[E2:B5 + 1, :])
        midi_matrix = overlap_check(instrument, midi_matrix)
        if midi_matrix is None:
            continue

        midi_matrix = one_hot(midi_matrix)
        attack_matrix = find_attack_matrix(midi_data, vector, midi_matrix, tempo)

        encoded_matrices.append(
            (np.append(midi_matrix,attack_matrix,axis=0), timesteps_passed)
        )
        timesteps_passed += midi_matrix.shape[1]

    return encoded_matrices

def decoding_to_midi(encoded_matrix, tempo=100, time_signature="4/4"):
    """
    Decodes a matrix encoded for LSTM back to a PrettyMIDI file
    :param encoded_matrix:
    :param tempo:
    :param time_signature:
    :return midi_data:
    """
    # Trim out beat matrix
    approx_to_zero = np.vectorize(lambda x: 1 if np.random.random() < x else 0)
    # Split encoded_matrix into piano_roll and attack_matrix
    piano_roll = approx_to_zero(encoded_matrix[:B5 - E2 + 1, :])
    attack_matrix = approx_to_zero(encoded_matrix[-6:, :])
    plucks_per_timestep = attack_matrix.sum(axis=0)
    pluck_nonzero = plucks_per_timestep.nonzero()[0]

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_signature = time_signature.split('/')
    midi_data.time_signature_changes.append(
        pretty_midi.TimeSignature(int(time_signature[0]), int(time_signature[1]), 0)
    )
    midi_data.instruments.append(pretty_midi.Instrument(24, name="Guitar"))

    timesteps_per_second = tempo / 60 * beat_length

    for timestep in reversed(pluck_nonzero):
        plucks = np.array(attack_matrix[:, timestep].nonzero(), dtype=np.int32).flatten()
        notes_played = np.array(piano_roll[:, timestep].nonzero(), dtype=np.int32).flatten()
        try:
            pitches = notes_played[plucks]
        except IndexError:
            continue
        for pitch in pitches:
            notes_equal_pitch = list(filter(lambda note: note.pitch == pitch + E2,
                                        midi_data.instruments[0].notes))
            if notes_equal_pitch:
                next_pluck = int(min(notes_equal_pitch,
                        key=lambda note: note.start).start \
                         * timesteps_per_second)
            else:
                next_pluck = 1e20
            note_timesteps = 1
            while timestep + note_timesteps < piano_roll.shape[1] \
                  and timestep + note_timesteps < next_pluck \
                    and piano_roll[pitch, timestep + note_timesteps]:
                note_timesteps += 1
            note_length = note_timesteps / timesteps_per_second
            begin = timestep / timesteps_per_second
            midi_data.instruments[0].notes.append(pretty_midi.Note(127, pitch + E2, begin, begin + note_length))

    return midi_data

def find_beat_matrix(midi_matrix, starting_timestep):
    """
    Returns a beat matrix corresponding to the midi_matrix
    TODO: Explain this more
    :param midi_matrix: generated midi_matrix
    :return: beat_matrix
    """

    timesteps = midi_matrix.shape[1]
    template_length = beat_template.shape[1]
    starting_timestep = starting_timestep % template_length
    beat_matrix = np.empty((beat_template.shape[0], 0))
    for i in range(int((timesteps + starting_timestep) / template_length) + 1):
        beat_matrix = np.append(beat_matrix, beat_template, axis=1)
    return beat_matrix[:, starting_timestep:timesteps + starting_timestep]

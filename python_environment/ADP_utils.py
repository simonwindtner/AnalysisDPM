import numpy as np
import tensorflow as tf
from mido import Message, MidiFile, MidiTrack, second2tick, MetaMessage, bpm2tempo
import dill
from ddsp_piano.utils.io_utils import load_midi_as_conditioning
from soundfile import write 

def save_as_wav(outputs, filename="out.wav"):
        data = outputs['audio_synth'][0].numpy()
        data /= np.abs(data).max()
        write(filename,
                data=data,
                samplerate=16000)
        print(f"Audio saved at {filename}")

def play(outputs, fs=16000):
    data = outputs['audio_synth'][0].numpy()
    Audio(data, rate=fs)


def amp_to_db(x):
  return 20*np.log10(x)

def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

def tf_float32(x):
  """Ensure array/tensor is a float32 tf.Tensor."""
  if isinstance(x, tf.Tensor):
    return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
  else:
    return tf.convert_to_tensor(x, tf.float32)
  
def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
  """Exponentiated Sigmoid pointwise nonlinearity.

  Bounds input to [threshold, max_value] with slope given by exponent.

  Args:
    x: Input tensor.
    exponent: In nonlinear regime (away from x=0), the output varies by this
      factor for every change of x by 1.0.
    max_value: Limiting value at x=inf.
    threshold: Limiting value at x=-inf. Stablizes training when outputs are
      pushed to 0.

  Returns:
    A tensor with pointwise nonlinearity applied.
  """
  x = tf_float32(x)
  return max_value * tf.nn.sigmoid(x)**tf.math.log(exponent) + threshold


def notename2midinote(notename):
    shift = 0
    length = len(notename)
    if length not in [2,3]:
        pass
    if length == 3:
        shiftdict = {"#": 1, "b":-1}
        shiftchar = notename[1]
        if shiftchar not in shiftdict:
            raise LookupError(f"{shiftchar} must be b or #")
        shift = shiftdict[shiftchar]
    noteind = ord(notename[0])
    if not (noteind >= 65 and noteind <= 71):
        raise LookupError("Note Name must be in [A,B,C,D,E,F,G]")
    octave = int(notename[length-1])
    notedist = [-12,2,1,2,2,1,2,2]
    return (sum(notedist[:(noteind%65)+1])) + 12 * octave + shift + 21

def create_simple_midi_file(notedict, outfile = "out.mid"):
    beats = notedict["beats"]
    velocities = notedict["velocities"]
    pedal = notedict["pedal"]
    pause = notedict["pause"]
    duration = notedict["duration"]
    silence = notedict["silence"]
    tempo = bpm2tempo(120)
    ticks_per_beat = 100
    mid = MidiFile(ticks_per_beat=ticks_per_beat, type=1)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo)) 
    track.append(Message("control_change", control=64, value=pedal, time=0))
    track.append(Message("control_change", control=67, value=127, time=0))
    
    for m, beat in enumerate(beats):
        commands = {0:"note_on", 1:"note_off"}
        sleep_ticks_dict = {0:pause, 1:duration}
        for i in range(2):
            for j, note in enumerate(beat):
                if j == 0:
                    sleep_ticks = second2tick(sleep_ticks_dict[i], ticks_per_beat=ticks_per_beat, tempo=tempo)
                else:
                    sleep_ticks = 0
                track.append(Message(commands[i], note=notename2midinote(note), velocity=velocities[m][j], time=sleep_ticks))
    track.append(Message("note_on", note=notename2midinote(note), velocity=1, time=second2tick(silence, ticks_per_beat=ticks_per_beat, tempo=tempo)))
    track.append(Message("note_off", note=notename2midinote(note), velocity=0, time=second2tick(0.1, ticks_per_beat=ticks_per_beat, tempo=tempo)))
    mid.save(outfile)

def create_simple_midi_file_note_numbers(notedict, outfile = "out.mid"):
    beats = notedict["beats"]
    velocities = notedict["velocities"]
    pedal = notedict["pedal"]
    pause = notedict["pause"]
    duration = notedict["duration"]
    silence = notedict["silence"]
    tempo = bpm2tempo(120)
    ticks_per_beat = 100
    mid = MidiFile(ticks_per_beat=ticks_per_beat, type=1)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo)) 
    track.append(Message("control_change", control=64, value=pedal, time=0))
    track.append(Message("control_change", control=67, value=127, time=0))
    
    for m, beat in enumerate(beats):
        commands = {0:"note_on", 1:"note_off"}
        sleep_ticks_dict = {0:pause, 1:duration}
        for i in range(2):
            for j, note in enumerate(beat):
                if j == 0:
                    sleep_ticks = second2tick(sleep_ticks_dict[i], ticks_per_beat=ticks_per_beat, tempo=tempo)
                else:
                    sleep_ticks = 0
                track.append(Message(commands[i], note=note, velocity=velocities[m][j], time=sleep_ticks))
    track.append(Message("note_on", note=note, velocity=1, time=second2tick(silence, ticks_per_beat=ticks_per_beat, tempo=tempo)))
    track.append(Message("note_off", note=note, velocity=0, time=second2tick(0.1, ticks_per_beat=ticks_per_beat, tempo=tempo)))
    mid.save(outfile)

def create_midi_and_synthesize(notedict, model, duration, piano_type, notenumber = False, f_vibrato = 0, amplitude_vibrato = 0, factor_velocity = 1, callargs = None, save_output=False, output_file="out.pkl"):
    midioutfile = "out2.mid"
    if notenumber:
        create_simple_midi_file_note_numbers(notedict, outfile = midioutfile)
    else:
        create_simple_midi_file(notedict, outfile = midioutfile)
    inputs = load_midi_as_conditioning(midioutfile, duration=duration)
    length_conditioning = inputs["conditioning"].shape[1]
    inputs["conditioning"][0,:,:,0] += amplitude_vibrato * np.tile(np.sin(np.arange(length_conditioning)*np.pi*f_vibrato/125), (16,1)).T
    # inputs["conditioning"][0,:,:,0][inputs["conditioning"][0,:,:,0] != 0] += amplitude_vibrato * np.tile(np.sin(np.arange(length_conditioning)*np.pi*f_vibrato/125), (16,1)).T[inputs["conditioning"][0,:,:,0] != 0]
    inputs["conditioning"][0,:,:,1] *= factor_velocity
    inputs['piano_model'] = tf.convert_to_tensor([[piano_type]])
    if callargs is not None:
        model.callargs = callargs
    output = model(inputs)
    if save_output:
        with open(output_file, "wb") as file:
            dill.dump(output, file)
    return output

def create_midi_and_synthesize_glissando(notedict, model, duration, piano_type, stepsize):
    outfile = "Glissando_demo_new.mid"
    create_simple_midi_file(notedict, outfile = outfile)
    inputs = load_midi_as_conditioning(outfile, duration=duration)
    start_idx = np.where(inputs['conditioning'][0,:,:,0] > 0)[0][0]
    length_conditioning = inputs["conditioning"].shape[1]
    start_note = inputs['conditioning'][0,start_idx,0,0]
    note_ascending = True
    for i in range(start_idx, length_conditioning, stepsize):
        inputs["conditioning"][0,i:i+stepsize,0,0] = np.linspace(start_note, start_note +1, stepsize,dtype=float)
        if note_ascending:
            start_note += 1
        else:
            start_note -= 1
        if start_note > 108:
            note_ascending = False
    inputs['piano_model'] = tf.convert_to_tensor([[piano_type]])
    return model(inputs)



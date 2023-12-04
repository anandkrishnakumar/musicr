import streamlit as st

import io
import math

import numpy as np
import tensorflow as tf
import pydub
import matplotlib.pyplot as plt
from audiorecorder import audiorecorder

from tensorflow.keras.models import load_model

@st.cache_resource
def load_models():
  model_eval = load_model('models/note_detect.h5', compile=False)
  return model_eval

def to_array(a, normalized=False):
    """PyDub to numpy array"""
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def freq_to_note(freq):
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    note_number = 12 * math.log2(freq / 440) + 49  
    note_number = round(note_number)
        
    note = (note_number - 1 ) % len(notes)
    note = notes[note]
    
    octave = (note_number + 8 ) // len(notes)
    
    return str(note), str(octave)

def get_note(a):
  print(a.shape)
  X = np.expand_dims(a, 0)

  fig, ax = plt.subplots()
  ax.plot(X[0])
  st.pyplot(fig)


  y = model_eval.predict(X)
  st.write(f"The note is {' '.join(freq_to_note(y))}")

st.set_page_config(
        page_title="MUSICR",
)

st.title("Note Detection")

st.write("Upload a file or Record")
audio_buffer = st.file_uploader("Upload an mp3 file", type=["mp3"])
audio = audiorecorder("Click to record", "Click to stop recording")

model_eval = load_models()
if st.button("Find note"):
    if audio_buffer is not None:
      audio_read = pydub.AudioSegment.from_file(audio_buffer, "mp3")
      a = to_array(audio_read, normalized=True)[1][:500, 0]
      get_note(a)
    elif len(audio) > 0:
      # To play audio in frontend:
      st.audio(audio.export().read())
      # To get audio properties, use pydub AudioSegment properties:
      st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
      a = to_array(audio, normalized=True)[1][400:400+500]
      a /= np.max(np.abs(a),axis=0)
      get_note(a)
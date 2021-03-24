
import numpy as np
import tensorflow as tf

def preprocess(audio_data, batch_size = 128, frame_size = 256, frame_shift=128):

  X = []
  Y = []

  audio_len = len(audio_data)
  for i in range(0, audio_len - frame_size - 1, frame_shift):
    frame = audio_data[i:i + frame_size]
    if len(frame) < frame_size:
      break
    if i + frame_size >= audio_len:
        break
    temp = audio_data[i + frame_size]
    target_val = int((np.sign(temp) * (np.log(1 + 256 * abs(temp)) / (np.log(1 + 256))) + 1) / 2.0 * 255)
    X.append(frame.reshape(frame_size, 1))
    Y.append((np.eye(256)[target_val]))
  fr = np.array(X), np.array(Y)
  ds = tf.data.Dataset.from_tensor_slices(fr)
  
  ds = ds.repeat()

  ds = ds.batch(batch_size)
  return ds
# t = create_dataset(training_data)
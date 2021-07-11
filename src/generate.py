


from scipy.io.wavfile import write
from tqdm import tqdm
from tensorflow import keras
import numpy as np
import glob
import os
import random
import datetime
import logging
import argparse

logger = logging.getLogger()


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

desc = "WaveNet"
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs for training')

parser.add_argument('--sr', type=int, default=16000, help='sample_rate')

parser.add_argument('--ns', type=int, default=2, help='num_samples')

parser.add_argument('--fsz', type=int, default=256, help='frame_size')

parser.add_argument('--mp', type=str, help='model path')

parser.add_argument('--nfls', type=str, default = 1, help='no. of files')
p = parser.parse_args()


p = parser.parse_args()
if p.mp == None:
    logger.error('model path must be specified')
    exit()

def generate_audio(model, sr, frame_size, no_files, generated_seconds, training_audio=None):
    au_arr = training_audio[:frame_size]
    for i in tqdm(range(no_files)):
        gen_audio = np.zeros((sr * generated_seconds))
        for curr_sample_idx in tqdm(range(gen_audio.shape[0])):
            distribution = np.array(model.predict(au_arr.reshape(1, frame_size, 1)), dtype=float).reshape(256)
            distribution /= distribution.sum().astype(float)
            predicted_val = np.random.choice(range(256), p=distribution)
            amplified_8 = predicted_val / 255.0
            amplified_16 = (np.sign(amplified_8) * (1/255.0) * ((1 + 256.0)**abs(amplified_8) - 1)) * 2**15
            gen_audio[curr_sample_idx] = amplified_16
            return gen_audio, training_audio, i


model = keras.models.load_model(p.mp)
def save_audio():  
    logger.info("Generating Audio.")
    gen_audio, training_audio, i = generate_audio(model, p.sr, p.fsz, p.ns, p.nfls, np.load('training_data.npy'))
    wavname = ( "_sample_1" + str(i) + '.wav')
    outputPath = 'generated'+'/'+ wavname
    logger.info("Saving File to " + outputPath)
    write(outputPath, p.sr, gen_audio.astype(np.int16))
    logger.info("Generating Audio.")
save_audio()
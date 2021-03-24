from future import *
from model import model
from utils import preprocess
import tensorflow
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import argparse
import logging
import datetime
import tensorboard
import glob
import librosa
import numpy as np
import random
import os
import tqdm


tensorflow.keras.backend.clear_session()
logger = logging.getLogger()

desc = "WaveNet"
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs for training')

parser.add_argument('--bs', type=int, default=128, help='batch size')

parser.add_argument('--isz', type=int, default=256, help='input_size')

parser.add_argument('--nl', type=int, default=40, help='num_layers')

parser.add_argument('--ks', type=int, default=2, help='kernel_size')

parser.add_argument('--dr', type=int, default=2, help='dilation_rate')

parser.add_argument('--nf', type=int, default=64, help='num_filters')

parser.add_argument('--sr', type=int, default=16000, help='sample_rate')

parser.add_argument('--ns', type=int, default=2, help='num_samples')

parser.add_argument('--fs', type=int, default=128, help='frame_shift')

parser.add_argument('--fsz', type=int, default=256, help='frame_size')

parser.add_argument('--dp', type=str, help='path_to_dataset')

parser.add_argument('--s', type=str, default = False, help='print summary')

parser.add_argument('--mp', type=str, help='model path')

parser.add_argument('--nfls', type=str, default = 1, help='no. of files')
p = parser.parse_args()


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )


if p.dp is None:
    
    logger.error('dataset path must be specified')
    exit()

lr_scheduler = ReduceLROnPlateau(patience=10/ 2, 
                                 cooldown=10 / 4,
                                 verbose=1)
        

earlystopping_callback = EarlyStopping(monitor='accuracy',
                                           min_delta=0.01,
                                           patience=10,
                                           verbose=0,
                                           restore_best_weights=True)

logdir = ".logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=logdir)


logger.info('Loading data..')

files = []
training_data = []

for filename in glob.glob(os.path.join(p.dp, '*.wav')):
    files.append(filename)

    random.shuffle(files)

if p.ns is not None:

    randomized_files = files[:p.ns]
else:
    logger.error("number of samples not specified")

for training_filename in tqdm.tqdm(randomized_files):

    try:
        audio, _ = librosa.load(training_filename, sr=p.sr, mono=True)
        print(audio)
        audio = audio.reshape(-1, 1)
        training_data = training_data + audio.tolist()
    except:
        pass

training_data = np.array(training_data)
training_audio_length = len(training_data)
np.save('data/training_data', training_data)
logger.info('preprocessing..')
training_data = preprocess.preprocess(training_data, p.bs, p.fsz, p.fs)
logger.info('**training started**')

model = model.WaveNet(p.isz, p.nl, p.ks, p.dr, p.nf).model()
model.compile(loss='categorical_crossentropy',
                  optimizer='adam')
if p.s == True:
    model.summary()

model.fit(training_data, 
              epochs=p.epochs,
              steps_per_epoch=training_audio_length // 128,
              verbose=1,callbacks=[earlystopping_callback, tensorboard_callback])
model.save('.trained_model/modelWN.h5')
logger.info("model saved in model/n Training finished successfully")
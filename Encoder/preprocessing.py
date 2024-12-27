import os
import glob
import librosa
import numpy as np

from params import hparams as hp

data_path = glob.glob(os.path.dirname(hp.unprocessed_data))

def preprocess_tisv_spectrogram():

    print("Starting preprocessing...")
    os.makedirs(hp.train_path, exist_ok=True)
    os.makedirs(hp.test_path, exist_ok=True)

    nb_speakers = len(data_path)
    print(f"Number of speakers: {nb_speakers}")

    nb_speakers_train = int(nb_speakers * hp.train_ratio)
    print(f"Number of speakers in train: {nb_speakers_train}")

    nb_speakers_test = nb_speakers - nb_speakers_train
    print(f"Number of speakers in test: {nb_speakers_test}")

    for i, speaker in enumerate(data_path):
        for j, chapter in enumerate(os.listdir(speaker)):
            chapter_path = os.path.join(speaker, chapter)
            for k, utterance in enumerate(os.listdir(chapter_path)):
                utterance_path = os.path.join(chapter_path, utterance)
                if utterance[-5:] == '.flac':
                    utter, sr = librosa.core.load(utterance_path, sr=hp.sr)
                    intervals = librosa.effects.split(utter, top_db=40)
                    for interval in intervals:
                        if (interval[1] - interval[0])
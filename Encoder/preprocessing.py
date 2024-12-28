import os
from tqdm import tqdm
import glob
import librosa
import numpy as np
from params import hparams as hp

data_path = glob.glob(os.path.dirname(hp.unprocessed_data))


def preprocess_tisv_spectrogram():

    print("Starting preprocessing...")

    os.makedirs(hp.data.train_path, exist_ok=True)
    os.makedirs(hp.data.test_path, exist_ok=True)

    nb_speakers = len(data_path)
    nb_speakers_train = int(nb_speakers * hp.train.ratio)
    nb_speakers_test = nb_speakers - nb_speakers_train

    print(f"Number of speakers: {nb_speakers}")
    print(f"Number of speakers in train: {nb_speakers_train}")
    print(f"Number of speakers in test: {nb_speakers_test}")

    for i, speaker in tqdm(enumerate(data_path)):
        utterances_spec = []
        for j, chapter in enumerate(os.listdir(speaker)):
            if chapter != '.DS_Store':
                chapter_path = os.path.join(speaker, chapter)
                for k, utterance in enumerate(os.listdir(chapter_path)):
                    utterance_path = os.path.join(chapter_path, utterance)
                    if utterance[-5:] == '.flac':
                        try:
                            utter, sr = librosa.core.load(utterance_path, sr=hp.data.sr) # noqa E501
                            intervals = librosa.effects.split(utter, top_db=40)
                            for interval in intervals:
                                utter_part = utter[interval[0]:interval[1]]
                                part_utter_duration = len(utter_part)/sr
                                if part_utter_duration > 1.6: # noqa E501
                                    nb_pot_utters = int(np.floor(part_utter_duration//1.6)) # noqa E501
                                    S = librosa.feature.melspectrogram(
                                        y=utter_part,
                                        sr=sr,
                                        n_fft=hp.data.nfft,
                                        hop_length=int(hp.data.hop*hp.data.sr),
                                        n_mels=hp.data.nmels
                                        )
                                    S = librosa.power_to_db(S, ref=np.max)
                                    for u in range(nb_pot_utters):
                                        utterances_spec.append(S[:, int(u*hp.data.tisv_frame):int((u+1)*hp.data.tisv_frame)]) # noqa E501
                                    utterances_spec.append(S[:, -hp.data.tisv_frame:]) # noqa E501
                        except Exception as e:
                            print(f'Error: {e} File {utterance_path} could not be loaded') # noqa E501
        if i < nb_speakers_train:
            np.save(os.path.join(hp.data.train_path, f"speaker_{i}.npy"), utterances_spec) # noqa E501
        else:
            np.save(os.path.join(hp.data.test_path, f"speaker_{i - nb_speakers_train}.npy"), utterances_spec) # noqa E501


if __name__ == '__main__':
    preprocess_tisv_spectrogram()

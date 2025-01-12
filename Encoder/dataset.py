import os
import torch
import random
import numpy as np
from params import hparams as hp
from torch.utils.data import Dataset


class EncoderDataset(Dataset):

    def __init__(self, train=True, shuffle=True, utter_start=0):
        self.train = train
        if self.train:
            self.data_path = hp.data.train_path
            self.nb_utterances = hp.train.M
        else:
            self.data_path = hp.data.test_path
            self.nb_utterances = hp.test.M
        self.speakers = os.listdir(self.data_path)
        self.utter_start = utter_start
        self.shuffle = shuffle

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):

        if self.shuffle:
            speaker = random.sample(self.speakers, 1)[0]
        else:
            speaker = self.speakers[idx]

        utterances = np.load(os.path.join(self.data_path, speaker))

        if self.shuffle:
            utter_index = np.random.randint(0, utterances.shape[0], self.nb_utterances) # noqa E501
            utterance = utterances[utter_index]
        else:
            utterance = utterances[self.utter_start: self.utter_start + self.nb_utterances] # noqa E501

        return torch.tensor(np.transpose(utterance, axes=(0, 2, 1)))

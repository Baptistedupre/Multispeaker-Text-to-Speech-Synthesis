import torch
import numpy as np
from text import text_to_sequence
from torch.utils.data import Dataset
from synthesizer_params import hparams as hp


class SynthesizerDataset(Dataset):
    def __init__(self, training=True):
        super(SynthesizerDataset, self).__init__()
        if training:
            data_path = hp.train.train_path
            self.data = np.load(data_path, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
    
        text = item['text']
        text = np.asarray(text_to_sequence(text, ['english_cleaners']), dtype=np.int32)
        text_length = len(text)
        pos_text = np.arange(1, text_length+1)

        mel = np.array(item['mel_spectrogram'], dtype=np.float32)
        mel_input = np.concatenate([np.zeros([1, hp.model.num_mels], np.float32), mel[:-1, :]], axis=0)
        pos_mel = np.arange(1, mel.shape[0] + 1)

        speaker_embedding = np.array(item['speaker_embedding'], dtype=np.float32) # noqa E501

        sample = {
            'text': text,
            'mel': mel, 
            'speaker_embedding': speaker_embedding,
            'text_length': text_length,
            'mel_input': mel_input,
            'pos_mel': pos_mel,
            'pos_text': pos_text
        }

        return sample


def synthesizer_collate_fn(batch):
    text = [d['text'] for d in batch]
    mel = [d['mel'] for d in batch]
    speaker_embedding = [d['speaker_embedding'] for d in batch]
    mel_input = [d['mel_input'] for d in batch]
    text_length = [d['text_length'] for d in batch]
    pos_mel = [d['pos_mel'] for d in batch]
    pos_text= [d['pos_text'] for d in batch]

    text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
    mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
    speaker_embedding = [i for i, _ in sorted(zip(speaker_embedding, text_length), key=lambda x: x[1], reverse=True)]
    mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
    pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
    pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
    text_length = sorted(text_length, reverse=True)

    text = _prepare_data(text).astype(np.int32)
    mel = _pad_mel(mel)
    mel_input = _pad_mel(mel_input)
    pos_mel = _prepare_data(pos_mel).astype(np.int32)
    pos_text = _prepare_data(pos_text).astype(np.int32)


    return {
        "text": torch.LongTensor(text),
        "mel": torch.FloatTensor(mel),
        "mel_input": torch.FloatTensor(mel_input),
        "speaker_embedding": torch.FloatTensor(np.array(speaker_embedding)),
        "pos_text": torch.LongTensor(pos_text),
        "pos_mel": torch.LongTensor(pos_mel),
        'text_length': torch.LongTensor(text_length)
    }


def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])
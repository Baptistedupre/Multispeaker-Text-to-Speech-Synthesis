import torch
import numpy as np
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
        mel_spectrogram = np.array(item['mel_spectrogram'], dtype=np.float32)
        speaker_embedding = np.array(item['speaker_embedding'], dtype=np.float32) # noqa E501
        return text, mel_spectrogram, speaker_embedding


def synthesizer_collate_fn(batch):
    texts, mels, speaker_embeddings = zip(*batch)

    text_lengths = torch.tensor([len(text) for text in texts], dtype=torch.long) # noqa E501
    max_text_len = max(text_lengths)
    text_padded = torch.zeros(len(texts), max_text_len, dtype=torch.long)

    for i, text in enumerate(texts):
        text_token_ids = torch.tensor([ord(c) for c in text], dtype=torch.long) # noqa E501
        text_padded[i, :len(text_token_ids)] = text_token_ids

    mel_lengths = torch.tensor([mel.shape[1] for mel in mels], dtype=torch.long) # noqa E501
    n_mels = mels[0].shape[0]
    max_mel_len = max(mel_lengths)
    mel_padded = torch.zeros(len(mels), n_mels, max_mel_len, dtype=torch.float32) # noqa E501

    for i, mel in enumerate(mels):
        mel_padded[i, :, :mel.shape[1]] = torch.tensor(mel, dtype=torch.float32) # noqa E501

    speaker_embeddings_tensor = torch.tensor(speaker_embeddings, dtype=torch.float32) # noqa E501 

    return {
        "text_padded": text_padded,
        "text_lengths": text_lengths,
        "mel_padded": mel_padded,
        "mel_lengths": mel_lengths,
        "speaker_embeddings": speaker_embeddings_tensor,
    }

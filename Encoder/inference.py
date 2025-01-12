import os
import torch
import librosa
import numpy as np

from Encoder.encoder_params import hparams as hp
from Encoder.model import SpeakerEncoder


def mel_spectogram(audio, sr, n_fft, hop, n_mels):

    mel = librosa.feature.melspectrogram(
                                        y=audio,
                                        sr=sr,
                                        n_fft=n_fft,
                                        hop_length=int(hop*sr),
                                        n_mels=n_mels
                                        )

    mel = librosa.power_to_db(mel, ref=np.max)
    return mel


def partial_utterances(waveform, sampling_rate, duration=1.6, overlap=0.5): # noqa E501
    num_samples = int(duration * sampling_rate)

    hop_length = int(num_samples * (1 - overlap))

    partial_utterances = []
    for start_idx in range(0, len(waveform) - num_samples + 1, hop_length):
        partial_utterance = waveform[start_idx:start_idx + num_samples]
        partial_utterances.append(partial_utterance)

    if len(waveform) % hop_length != 0:
        last_partial = waveform[-num_samples:]
        partial_utterances.append(last_partial)

    return partial_utterances


def embed_utterance(wav, model, device):

    intervals = librosa.effects.split(wav, top_db=40)
    utter = np.concatenate([wav[interval[0]:interval[1]] for interval in intervals], axis=0) # noqa E501
    part_utter = partial_utterances(utter, hp.data.sr)

    batch = []
    for utter in part_utter:
        batch.append(mel_spectogram(utter, hp.data.sr, hp.data.nfft, hp.data.hop, hp.data.nmels)) # noqa E501

    batch = np.array(batch)
    frames = torch.from_numpy(batch).to(device)
    frames = frames.transpose(1, 2)
    partial_embeds = model(frames).detach().cpu().numpy()
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    return embed


if __name__ == "__main__":
    model = SpeakerEncoder()
    model.load_state_dict(torch.load(os.path.join(hp.model.model_path, 'model_final.pt'), weights_only=True)) # noqa E501
    model.eval()
    wav, _ = librosa.core.load('/Users/bapt/Desktop/ENSAE/3ème Année/Advanced Machine Learning/Datasets/LibriSpeech/train-clean-360/14/208/14-208-0000.flac', sr=hp.data.sr) # noqa E501
    print(embed_utterance(wav))

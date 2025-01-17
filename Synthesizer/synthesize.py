import os
import torch
import librosa
from scipy.io.wavfile import write
from text import text_to_sequence
import numpy as np
from model.model import TransformerTTS
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import soundfile as sf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint():
    state_dict = torch.load('/Users/hifat/OneDrive/Bureau/AML Project/Saved Models/Synthesizer/model_2.pt')   
    new_state_dict = OrderedDict()
    for k, value in state_dict.items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict


def synthesis(text, speaker_embedding, max_len):
    model = TransformerTTS().to(device)

    model.load_state_dict(torch.load('/Users/hifat/OneDrive/Bureau/AML Project/Saved Models/Synthesizer/model_50epochs.pt', weights_only=True))
    model.eval()

    text = np.asarray(text_to_sequence(text, ['english_cleaners']))
    text = torch.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    mel_input = torch.zeros([1, 1, 80]).cuda()
    pos_text = torch.arange(1, text.size(1)+1).unsqueeze(0)
    pos_text = pos_text.cuda()

    speaker_embedding = torch.FloatTensor(np.array(speaker_embedding)).unsqueeze(0).cuda()
    
    pbar = tqdm(range(max_len))
    with torch.no_grad():
        for i in pbar:
            pos_mel = torch.arange(1,mel_input.size(1)+1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = model.forward(text, pos_text, mel_input, pos_mel, speaker_embedding)
            mel_input = torch.cat([mel_input, mel_pred[:, -1:, :]], dim=1)
        
    mel_out = postnet_pred.squeeze().detach().cpu().numpy().T
    
    mel_spectrogram = librosa.db_to_amplitude(mel_out)

    sr = 16000
    n_fft = 2048
    hop_length = 256
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=mel_spectrogram.shape[0])
    inv_mel_spectrogram = np.dot(np.linalg.pinv(mel_basis), mel_spectrogram)

    audio = librosa.griffinlim(inv_mel_spectrogram, hop_length=hop_length)

    sf.write('/Users/hifat/OneDrive/Bureau/AML Project/Datasets/Synthesizer/audio.wav', audio , sr)

    plt.figure(figsize=(10, 4))
    plt.imshow(
        mel_out,
        aspect='auto',
        origin='lower',
        interpolation='none',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()
    

    

if __name__ == '__main__':
    speaker_embedding = np.load('/Users/hifat/OneDrive/Bureau/AML Project/Datasets/Synthesizer/test_data_bon_sr_golmon.npy', allow_pickle=True)[3]['mel_spectrogram'].transpose()
    #Synthesis("ON THE RIGHT BANK OF THE AUFIDUS THE NEXT MORNING EMILIUS WHO WAS IN COMMAND DETACHED A THIRD OF HIS FORCE ACROSS THE RIVER AND ENCAMPED THEM THERE FOR THE PURPOSE OF SUPPORTING THE ROMAN FORAGING PARTIES ON THAT SIDE AND OF INTERRUPTING THOSE OF THE CARTHAGINIANS", speaker_embedding, 400)
    plt.figure(figsize=(10, 4))
    plt.imshow(
        speaker_embedding,
        aspect='auto',
        origin='lower',
        interpolation='none',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()
    #train_loss = np.load('/Users/hifat/OneDrive/Bureau/AML Project/Saved Models/Synthesizer/train_loss_high_res_6layers.npy', allow_pickle=True)

    #plt.plot(train_loss)
    #plt.show()
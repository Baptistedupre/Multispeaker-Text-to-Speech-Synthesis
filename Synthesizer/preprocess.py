import os
from tqdm import tqdm
import glob
import librosa
import numpy as np
from params import hparams as hp
from encoder import SpeakerEncoder
from torchaudio import transforms

DATASET_PATH = "path/to/trained_clean_360"

speaker_encoder = SpeakerEncoder()
speaker_encoder.eval()

def mel_spectogram(
        sample_rate,
        hop_length,
        win_length,
        n_fft,
        n_mels,
        f_min,
        f_max,
        power,
        normalized,
        norm,
        mel_scale,
        compression,
        audio,
):

    audio_to_mel = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=power,
        normalized=normalized,
        norm=norm,
        mel_scale=mel_scale,
    ).to(audio.device)

    mel = audio_to_mel(audio)
    return mel

def process_audio(audio_path):
    # Load audio
    waveform, sr = librosa.load(audio_path, sr=None)
    
    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return waveform, mel_spectrogram

def process_transcripts(transcript_path):
    # Parse the transcript file
    with open(transcript_path, "r") as file:
        lines = file.readlines()
    transcripts = {}
    for line in lines:
        utterance_id, text = line.split(" ", 1)
        transcripts[utterance_id] = text.strip()
    return transcripts

def process_speaker_folder(speaker_path):
    dataset = []
    for chapter in os.listdir(speaker_path):
        chapter_path = os.path.join(speaker_path, chapter)
        if not os.path.isdir(chapter_path):
            continue
        
        # Load transcripts
        transcript_file = os.path.join(chapter_path, f"{chapter}.txt")
        transcripts = process_transcripts(transcript_file)
        
        for audio_file in os.listdir(chapter_path):
            if audio_file.endswith(".flac"):
                audio_path = os.path.join(chapter_path, audio_file)
                utterance_id = os.path.splitext(audio_file)[0]
                
                # Process audio
                waveform, mel_spectrogram = process_audio(audio_path)
                
                # Compute speaker embedding
                speaker_embedding = speaker_encoder.infer(waveform)
                
                # Append to dataset
                dataset.append({
                    "utterance_id": utterance_id,
                    "text": transcripts.get(utterance_id, ""),
                    "mel_spectrogram": mel_spectrogram.tolist(),
                    "speaker_embedding": speaker_embedding.tolist()
                })
    return dataset

# Process all speakers
all_data = []
for speaker in os.listdir(DATASET_PATH):
    speaker_path = os.path.join(DATASET_PATH, speaker)
    if os.path.isdir(speaker_path):
        all_data.extend(process_speaker_folder(speaker_path))

# Save to JSON (or other format)
import json
with open("processed_librispeech_clean360.json", "w") as json_file:
    json.dump(all_data, json_file)

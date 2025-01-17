import os
import sys
import torch
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from Encoder.model import SpeakerEncoder # noqa E501
from Encoder.inference import embed_utterance # noqa E501
from Encoder.encoder_params import hparams as hp # noqa E501

target_folder = 'Datasets/Synthesizer' # noqa E501
device = torch.device("cuda" if torch.cuda().is_available() else "cpu")
model = SpeakerEncoder().to(device)
model.load_state_dict(torch.load(os.path.join(hp.model.model_path, 'model_final.pt'), weights_only=True)) # noqa E501


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


def process_transcripts(transcript_path):
    # Parse the transcript file
    with open(transcript_path, "r") as file:
        lines = file.readlines()
    transcripts = {}
    for line in lines:
        utterance_id, text = line.split(" ", 1)
        transcripts[utterance_id] = text.strip()
    return transcripts


def process_speaker_folder(speaker_path, speaker_embeddings):
    dataset = []
    speaker = os.path.basename(speaker_path)
    for chapter in os.listdir(speaker_path):
        chapter_path = os.path.join(speaker_path, chapter)
        if not os.path.isdir(chapter_path):
            continue

        transcript_file = os.path.join(chapter_path, f"{speaker}-{chapter}.trans.txt") # noqa E501
        transcripts = process_transcripts(transcript_file)

        for audio_file in os.listdir(chapter_path):
            if audio_file.endswith(".flac"):
                try:
                    audio_path = os.path.join(chapter_path, audio_file)
                    utterance_id = os.path.splitext(audio_file)[0]

                    audio, _ = librosa.core.load(audio_path, sr=16000)
                    intervals = librosa.effects.split(audio, top_db=40)
                    utter = np.concatenate([audio[interval[0]:interval[1]] for interval in intervals], axis=0) # noqa E501

                    mel_spectrogram = mel_spectogram(utter, 16000, 512, 0.01, 40) # noqa E501

                    dataset.append({
                        "text": transcripts[utterance_id],
                        "mel_spectrogram": mel_spectrogram,
                        "speaker_embedding": speaker_embeddings[speaker]
                    })
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")

    return dataset


def process_all_speakers(dataset_path, nb_speaker, speaker_embeddings):
    all_data = []
    for i, speaker in tqdm(enumerate(os.listdir(dataset_path))):
        speaker_path = os.path.join(dataset_path, speaker)
        if i <= nb_speaker:
            if os.path.isdir(speaker_path):
                all_data.extend(process_speaker_folder(speaker_path, speaker_embeddings)) # noqa E501
    return all_data


def compute_speakers_embedding(dataset_path, model, device): # noqa E501
    speaker_folders = [
        os.path.join(dataset_path, speaker)
        for speaker in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, speaker))
    ]

    embeddings = {}

    # Process speakers in batches
    with tqdm(total=len(speaker_folders), desc="Computing Embeddings") as pbar:
        for speaker_path in speaker_folders:
            speaker = os.path.basename(speaker_path)
            chapters = [
                os.path.join(speaker_path, chapter)
                for chapter in os.listdir(speaker_path)
                if os.path.isdir(os.path.join(speaker_path, chapter))
            ]
            if not chapters:
                continue
            selected_chapter = chapters[0]

            utterances = [
                os.path.join(selected_chapter, file)
                for file in os.listdir(selected_chapter)
                if file.endswith(".flac")
            ]
            if not utterances:
                continue
            selected_utterance = utterances[1]

            # Load the audio and compute the mel spectrogram
            audio, _ = librosa.load(selected_utterance, sr=None)
            embedding = embed_utterance(audio, model, device)
            embeddings[speaker] = embedding

            pbar.update(1)

    return embeddings


if __name__ == "__main__":
    # embeddings = compute_speakers_embedding(DATASET_PATH, model, device)
    # np.save(os.path.join(target_folder, "speaker_embeddings.npy"), embeddings) # noqa E501
    embeddings = np.load(os.path.join(target_folder, "speaker_embeddings.npy"), allow_pickle=True).item() # noqa E501
    complete_dataset = process_all_speakers(DATASET_PATH, 500, embeddings)
    train_data, test_data = train_test_split(complete_dataset, test_size=0.1)
    train_path = os.path.join(target_folder, "train_data.npy")
    test_path = os.path.join(target_folder, "test_data.npy")
    np.save(train_path, train_data)
    np.save(test_path, test_data)

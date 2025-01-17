# Multispeaker-Text-to-Speech-Synthesis

# Transfer Learning from Speaker Verification to Text-to-Speech

## Overview
This project explores the use of transfer learning from speaker verification to Text-to-Speech (TTS) synthesis. Traditional TTS models require extensive audio data from a single speaker for training, which limits their ability to generalize to unseen voices. To address this, we integrate a pre-trained speaker encoder that enables our model to synthesize speech in the voice of different speakers using only a few seconds of their audio.

Our architecture is inspired by the Speaker Verification to Text-to-Speech (SV2TTS) framework, which allows for voice cloning with minimal data. This is achieved through a three-step process:
1. **Speaker Encoder**: Computes a speaker embedding from short utterances, following the methodology introduced in the Generalized End-to-End (GE2E) loss framework.
2. **Text-to-Speech Synthesizer**: Uses the speaker embedding as an additional input to generate mel spectrograms from text.

Our implementation utilizes a **TransformerTTS** synthesizer instead of the traditional Tacotron-based models, leveraging attention mechanisms for improved efficiency and performance. This repository doesn't include the vocoder part yet.

---

## Model Architecture
### 1. Speaker Encoder
- Implemented using a **3-layer LSTM** with a projection layer.
- Outputs **L2-normalized** embeddings of size **64** (reduced from 256 for computational feasibility).
- Trained using the **Generalized End-to-End Loss**, ensuring embeddings of similar voices are close in the latent space.

### 2. Synthesizer (TransformerTTS)
- Built upon the **Transformer** architecture for parallelized computations and better handling of long dependencies.
- Speaker embeddings are concatenated after the **encoder pre-net** and the positional encoding stage, allowing the model to process speaker identity from the start.

---

## Dataset
We use the **LibriSpeech Clean 360** dataset, which consists of **360 hours of speech from ~1000 speakers**. This provides the necessary diversity to train a multi-speaker TTS system.

- **Speaker Encoder Training**: Inputs are 40-channel log-mel spectrograms with a 25ms window and 10ms step.
- **Synthesizer Training**: Pairs of (text, mel spectrogram) generated from the dataset with 80-channels log-mel spectrogram.

---

## Future Work
- Implement a **trainable vocoder** (e.g., WaveNet, WaveRNN) to enhance audio quality.
- Experiment with **larger datasets** to improve speaker generalization.
- Optimize **synthesizer hyperparameters** for faster convergence and better synthesis quality.

---

## Installation & Usage
### Dependencies
Ensure you have the following dependencies installed:
```bash
pip install torch torchaudio numpy librosa
```

### Running the Model
1. **Train the Speaker Encoder**:
   ```bash
   python Encoder.train.py
   ```
2. **Train the Synthesizer**:
   ```bash
   python Synthesizer.train.py
   ```

---

## Repository Inspirations
This repository is implemented from scratch but was inspired by several existing projects that have contributed significantly to Text-to-Speech synthesis and speaker verification research. These include:

- [Transformer-TTS by choiHkk](https://github.com/choiHkk/Transformer-TTS): Provides a Transformer-based TTS model that enhances synthesis efficiency.
- [Transformer-TTS by soobinseo](https://github.com/soobinseo/Transformer-TTS): Another implementation of Transformer-based TTS, serving as a reference for our model.
- [Real-Time Voice Cloning by CorentinJ](https://github.com/CorentinJ/Real-Time-Voice-Cloning): Implements the SV2TTS framework for speaker adaptation using few-shot learning.
- [PyTorch Speaker Verification by HarryVolek](https://github.com/HarryVolek/PyTorch_Speaker_Verification): A reference implementation for training a speaker encoder using the GE2E loss.

These repositories provided valuable insights into the architectural choices and training strategies used in our model.

--- 

## References
- [SV2TTS Paper](https://arxiv.org/abs/1806.04558)
- [TransformerTTS Paper](https://arxiv.org/abs/2005.10380)
- [Generalized End-to-End Loss](https://arxiv.org/abs/1710.10467)



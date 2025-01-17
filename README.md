# Multispeaker-Text-to-Speech-Synthesis

# Transfer Learning from Speaker Verification to Text-to-Speech

## Overview
This project explores the use of transfer learning from speaker verification to Text-to-Speech (TTS) synthesis. Traditional TTS models require extensive audio data from a single speaker for training, which limits their ability to generalize to unseen voices. To address this, we integrate a pre-trained speaker encoder that enables our model to synthesize speech in the voice of different speakers using only a few seconds of their audio.

Our architecture is inspired by the Speaker Verification to Text-to-Speech (SV2TTS) framework, which allows for voice cloning with minimal data. This is achieved through a three-step process:
1. **Speaker Encoder**: Computes a speaker embedding from short utterances, following the methodology introduced in the Generalized End-to-End (GE2E) loss framework.
2. **Text-to-Speech Synthesizer**: Uses the speaker embedding as an additional input to generate mel spectrograms from text.
3. **Vocoder**: Converts the mel spectrograms into audio waveforms.

Our implementation utilizes a **TransformerTTS** synthesizer instead of the traditional Tacotron-based models, leveraging attention mechanisms for improved efficiency and performance.

---

## Model Architecture
### 1. Speaker Encoder
- Implemented using a **3-layer LSTM** with a projection layer.
- Outputs **L2-normalized** embeddings of size **64** (reduced from 256 for computational feasibility).
- Trained using the **Generalized End-to-End Loss**, ensuring embeddings of similar voices are close in the latent space.

### 2. Synthesizer (TransformerTTS)
- Built upon the **Transformer** architecture for parallelized computations and better handling of long dependencies.
- Incorporates **Multi-Head Attention** and **Scaled Dot-Product Attention**.
- Speaker embeddings are concatenated at the **encoder pre-net** stage, allowing the model to process speaker identity from the start.

### 3. Vocoder
- Instead of training a vocoder from scratch, we use **Librosa** for waveform reconstruction.
- Future improvements may involve training a **WaveRNN or WaveNet** vocoder.

---

## Dataset
We use the **LibriSpeech Clean 360** dataset, which consists of **360 hours of speech from ~1000 speakers**. This provides the necessary diversity to train a multi-speaker TTS system.

- **Speaker Encoder Training**: Inputs are 40-channel log-mel spectrograms with a 25ms window and 10ms step.
- **Synthesizer Training**: Pairs of (text, mel spectrogram) generated from the dataset.

---

## Training Details
- Speaker Encoder trained for **500 epochs (~200k iterations)**.
- TransformerTTS trained with **3 encoder layers and 3 decoder layers**.
- Reduced mel spectrogram dimensionality to **40 channels** instead of 80 for efficiency.
- Training conducted on a **single NVIDIA 1070Ti**, necessitating optimizations for computational feasibility.

### Speaker Verification Evaluation
- **Equal Error Rate (EER)** used to assess speaker embeddings.
- Enrollment tested with **1 to 5 utterances per speaker**.
- Achieved **EER of 4.32% with 3 enrolled utterances**, comparable to SV2TTS benchmarks.

### Synthesizer Evaluation
- Initial training revealed **overfitting issues**, leading to an increased number of encoder/decoder layers.
- Loss stabilized after adjustments, yielding **coherent synthesized speech** with distinct speaker characteristics.

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
   python train_speaker_encoder.py
   ```
2. **Train the Synthesizer**:
   ```bash
   python train_synthesizer.py
   ```
3. **Generate Speech**:
   ```bash
   python synthesize.py --text "Hello, world!" --speaker_audio sample.wav
   ```

---

## References
- [SV2TTS Paper](https://arxiv.org/abs/1806.04558)
- [TransformerTTS Paper](https://arxiv.org/abs/2005.10380)
- [Generalized End-to-End Loss](https://arxiv.org/abs/1710.10467)


import torch
from tqdm import tqdm
import torch.optim as optim
from dataset import SynthesizerDataset, synthesizer_collate_fn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.model import TransformerTTS
from model.loss import TransformerTTSLoss
from synthesizer_params import hparams as hp


def train_synthesizer(num_epochs, save_path, batch_size, log_interval=10):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TransformerTTS().to(device)
    criterion = TransformerTTSLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5) # noqa E501

    synthesizer_dataset = SynthesizerDataset(hp.train.train_path)
    train_size = int(0.9 * len(synthesizer_dataset))
    val_size = len(synthesizer_dataset) - train_size
    train_dataset, val_dataset = random_split(synthesizer_dataset, [train_size, val_size]) # noqa E501

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=synthesizer_collate_fn) # noqa E501
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=synthesizer_collate_fn) # noqa E501

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()

        train_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            optimizer.zero_grad()
            text = batch["text_padded"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            mel = batch["mel_padded"].to(device)
            mel_lengths = batch["mel_lengths"].to(device)
            speaker_embeddings = batch["speaker_embeddings"].to(device)
            output = model(text, text_lengths, mel, mel_lengths, speaker_embeddings) # noqa E501
            model_outputs = output[0], output[1], output[2]

            batch_size, _, mel_length = mel.size()
            gate_target = torch.zeros((batch_size, mel_length), device=device)
            for i, mel_len in enumerate(mel_lengths):
                gate_target[i, mel_len - 1:] = 1.0

            loss = criterion(model_outputs, (mel, gate_target))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % log_interval == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}") # noqa E501

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                text = batch["text_padded"].to(device)
                text_lengths = batch["text_lengths"].to(device)
                mel = batch["mel_padded"].to(device)
                mel_lengths = batch["mel_lengths"].to(device)
                speaker_embeddings = batch["speaker_embeddings"].to(device)
                output = model(text, text_lengths, mel, mel_lengths, speaker_embeddings) # noqa E501
                model_outputs = output[0], output[1], output[2]

                batch_size, _, mel_length = mel.size()
                gate_target = torch.zeros((batch_size, mel_length), device=device) # noqa E501
                for i, mel_len in enumerate(mel_lengths):
                    gate_target[i, mel_len - 1:] = 1.0

                loss = criterion(model_outputs, (mel, gate_target)) # noqa E501
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

        # Save the model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path} (Best validation loss: {best_val_loss:.4f})") # noqa E501

        # Update the learning rate scheduler
        scheduler.step(avg_val_loss)

    print("Training completed!")


if __name__ == "__main__":
    save_path = '/Users/bapt/Desktop/ENSAE/3ème Année/Advanced Machine Learning/Models/Synthesizer/transformer_tts.pt' # noqa E501
    train_synthesizer(num_epochs=50, save_path=save_path, batch_size=16)

import torch
from tqdm import tqdm
import torch.nn as nn 
import numpy as np
import torch.optim as optim
from dataset import SynthesizerDataset, synthesizer_collate_fn
from torch.utils.data import DataLoader, random_split

from model.model import TransformerTTS
from model.loss import TransformerTTSLoss
from synthesizer_params import hparams as hp
import matplotlib.pyplot as plt

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.train.lr * warmup_step**0.5 * min(step_num*warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_synthesizer(num_epochs, save_path, batch_size, log_interval=300):
    global_step = 0
    device = "cuda"
    model = TransformerTTS().to(device)
    criterion = TransformerTTSLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.train.lr)

    synthesizer_dataset = SynthesizerDataset(hp.train.train_path)
    train_size = int(0.9 * len(synthesizer_dataset))
    val_size = len(synthesizer_dataset) - train_size
    train_dataset, val_dataset = random_split(synthesizer_dataset, [train_size, val_size]) # noqa E501

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=synthesizer_collate_fn) # noqa E501
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=synthesizer_collate_fn) # noqa E501

    best_val_loss = float('inf')

    train_loss, val_loss = [], []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()

        train_loss_tot = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            global_step += 1

            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)

            text = batch["text"].to(device)
            mel = batch["mel"].to(device)
            mel_input = batch["mel_input"].to(device)
            speaker_embedding = batch["speaker_embedding"].to(device)
            pos_text = batch["pos_text"].to(device)
            pos_mel = batch["pos_mel"].to(device)
            stop_tokens = torch.abs(pos_mel.ne(0).type(torch.float) - 1)

            mel_pred, mel_postnet_pred, stop_pred, attn_enc, self_attn, dot_attn = model(text, pos_text, mel_input, pos_mel, speaker_embedding) # noqa E501

            loss = nn.L1Loss()(mel_pred, mel) + nn.L1Loss()(mel_postnet_pred, mel)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            train_loss.append(loss.item())
            train_loss_tot += loss.item()
            if batch_idx % log_interval == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                plt.figure(figsize=(10, 4))
                plt.imshow(
                    mel[0].detach().cpu().numpy().T,
                    aspect='auto',
                    origin='lower',
                    interpolation='none',
                    cmap='viridis'
                    )
                plt.colorbar(format='%+2.0f dB')
                plt.title('Actual mel spectrogram')
                plt.tight_layout()
                plt.savefig(f'/Users/hifat/OneDrive/Bureau/AML Project/Saved Models/Synthesizer/actual_mel_{epoch}_{batch_idx}_last_resort.png') # noqa E501
                plt.close()
                plt.figure(figsize=(10, 4))
                plt.imshow(
                    mel_pred[0].detach().cpu().numpy().T,
                    aspect='auto',
                    origin='lower',
                    interpolation='none',
                    cmap='viridis'
                    )
                plt.colorbar(format='%+2.0f dB')
                plt.title('Pred mel spectrogram')
                plt.tight_layout()
                plt.savefig(f'/Users/hifat/OneDrive/Bureau/AML Project/Saved Models/Synthesizer/pred_mel_{epoch}_{batch_idx}_last_resort.png') # noqa E501
                plt.close()

        avg_train_loss = train_loss_tot / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}") # noqa E501

        # Validation
        model.eval()
        val_loss_tot = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                text = batch["text"].to(device)
                mel = batch["mel"].to(device)
                mel_input = batch["mel_input"].to(device)
                speaker_embedding = batch["speaker_embedding"].to(device)
                pos_text = batch["pos_text"].to(device)
                pos_mel = batch["pos_mel"].to(device)
                stop_tokens = torch.abs(pos_mel.ne(0).type(torch.float) - 1)

                mel_pred, mel_postnet_pred, stop_pred, attn_enc, self_attn, dot_attn = model(text, pos_text, mel_input, pos_mel, speaker_embedding) # noqa E501
                loss = nn.L1Loss()(mel_pred, mel) + nn.L1Loss()(mel_postnet_pred, mel)

                val_loss.append(loss.item())
                val_loss_tot += loss.item()

        avg_val_loss = val_loss_tot / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

        # Save the model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path} (Best validation loss: {best_val_loss:.4f})") # noqa E501

        np.save('/Users/hifat/OneDrive/Bureau/AML Project/Saved Models/Synthesizer/train_loss_last_resort.npy', train_loss)
        np.save('/Users/hifat/OneDrive/Bureau/AML Project/Saved Models/Synthesizer/val_loss_last_resort.npy', val_loss)

    print("Training completed!")
    

if __name__ == "__main__":
    save_path = '/Users/hifat/OneDrive/Bureau/AML Project/Saved Models/Synthesizer/model_last_resort.pt' # noqa E501
    train_synthesizer(num_epochs=50, save_path=save_path, batch_size=8)

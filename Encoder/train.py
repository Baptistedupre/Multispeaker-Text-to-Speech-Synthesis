import os
import time
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from loss import GE2ELoss
from model import SpeakerEncoder
from encoder_params import hparams as hp
from utils import shuffle, unshuffle
from dataset import EncoderDataset
from visualization import save_umap


def train():

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu") # noqa E501
    print(device)
    
    train_dataset = EncoderDataset(train=True,
                                   shuffle=True)
    test_dataset = EncoderDataset(train=False,
                                  shuffle=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=hp.train.N,
                                  shuffle=True,
                                  num_workers=hp.train.num_workers,
                                  drop_last=True)

    model = SpeakerEncoder().to(device)
    ge2e_loss = GE2ELoss(device)

    if hp.train.restore:
        model.load_state_dict(torch.load(os.path.join(hp.train.checkpoint_dir, 'checkpoint.pt'), weights_only=True)['model_state'])

    optimizer = torch.optim.SGD([
                {'params': model.parameters()},
                {'params': ge2e_loss.parameters()}
            ], lr=hp.train.lr)

    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)

    train_loss = []
    iteration = 0
    for epoch in range(hp.train.epochs):
        total_loss = 0
        model.train()
        for batch_id, batch in tqdm(enumerate(train_dataloader, 1)):

            batch = batch.to(device)
            batch = torch.reshape(batch, (hp.train.N * hp.train.M, batch.size(2), batch.size(3))) # noqa E501

            perm = random.sample(range(0, hp.train.N * hp.train.M), hp.train.N * hp.train.M) # noqa E501
            batch = shuffle(batch, perm)

            optimizer.zero_grad()

            embeddings = model(batch)
            embeddings = unshuffle(embeddings, perm)
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1))) # noqa E501

            loss = ge2e_loss(embeddings)
            loss.backward()
            clip_grad_norm_(model.parameters(), 3.0)
            clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            iteration += 1
            train_loss.append(total_loss/(batch_id+1))

            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(), epoch+1, # noqa E501
                        batch_id+1, len(train_dataset)//hp.train.N, iteration,loss, total_loss / (batch_id + 1)) # noqa E501
                print(mesg)

            
        if epoch % hp.train.umap_interval == 0:
            speakers = random.sample(list(range(len(test_dataset))), 10)
            umap_embeddings = []
            for speaker in speakers:
                umap_embeddings += (model(test_dataset[speaker].to(device)).tolist())
            save_umap(umap_embeddings, epoch, hp.train.plot_dir)
            plt.close()

        print(f"Epoch: {epoch + 1} Loss: {total_loss}")
        torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, os.path.join(hp.train.checkpoint_dir, 'checkpoint_retrain_2000.pt'))
        np.save('/Users/hifat/OneDrive/Bureau/AML Project/Saved Models/Encoder/train_loss_2000.npy', train_loss)

    save_model_path = os.path.join(hp.model.model_path, "model_retrain.pt")
    torch.save(model.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
    train()

import os
import umap
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from encoder_params import hparams as hp
from model import SpeakerEncoder
from dataset import EncoderDataset


colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=float) / 255


def plot_umap(embeddings):
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(embeddings)

    ground_truth = np.repeat(np.arange(10), 10)
    colors = [colormap[i] for i in ground_truth]

    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors)
    plt.title(f"Embeddings UMAP projection")
    plt.show()


def save_umap(embeddings, epoch, save_path):
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(embeddings)

    ground_truth = np.repeat(np.arange(10), 10)
    colors = [colormap[i] for i in ground_truth]

    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors)
    plt.title(f"Embeddings UMAP projection - Epoch {epoch}")
    plt.savefig(f"C:/Users/hifat/OneDrive/Bureau/AML Project/Saved Models/Encoder/plots/epoch{epoch}.png")

if __name__ == "__main__":
    test_dataset = EncoderDataset(train=False,
                                  shuffle=False)
    speakers = random.sample(list(range(len(test_dataset))), 10)
    model = SpeakerEncoder()
    save_model_path = os.path.join(hp.train.checkpoint_dir, "checkpoint_retrain.pt")
    state_dict = torch.load(save_model_path, map_location=torch.device(hp.device), weights_only=True)['model_state'] # noqa E501
    model.load_state_dict(state_dict)
    embeddings = []
    for speaker in speakers:
        embeddings += (model(test_dataset[speaker]).tolist())
    plot_umap(embeddings)

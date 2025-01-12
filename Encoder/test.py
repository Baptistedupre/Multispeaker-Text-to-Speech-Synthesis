import os
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import SpeakerEncoder
from params import hparams as hp
from utils import shuffle, unshuffle, similarity_matrix_centroids
from Encoder.dataset import EncoderDataset


def test():

    device = torch.device(hp.device if torch.backends.mps.is_built() else "cpu") # noqa E501

    test_dataset = EncoderDataset(train=False,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=hp.test.N,
                                 shuffle=True,
                                 num_workers=hp.train.num_workers,
                                 drop_last=True)

    model = SpeakerEncoder().to(device)
    model.load_state_dict(torch.load(os.path.join(hp.model.model_path, 'model_final.pt'), weights_only=True)) # noqa E501
    model.eval()

    avg_eer = 0
    for epoch in tqdm(range(hp.test.epochs)):
        batch_avg_eer = 0
        for batch_id, batch in enumerate(test_dataloader, 1):
            batch = batch.to(device)

            nb_enroll = int(hp.test.M*hp.test.enroll_ratio)
            nb_verif = hp.test.M - nb_enroll

            enrollment_batch, verification_batch = torch.split(batch, [nb_enroll, nb_verif], dim=1) # noqa E501

            enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*enrollment_batch.size(1), enrollment_batch.size(2), enrollment_batch.size(3))) # noqa E501
            verification_batch = torch.reshape(verification_batch, (hp.test.N*verification_batch.size(1), verification_batch.size(2), verification_batch.size(3))) # noqa E501

            perm = random.sample(range(0, hp.test.N*nb_verif), hp.test.N*nb_verif) # noqa E501

            verification_batch = shuffle(verification_batch, perm)

            enrollment_embeddings = model(enrollment_batch)
            verification_embeddings = unshuffle(model(verification_batch), perm) # noqa E501

            enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, nb_enroll, enrollment_embeddings.size(1))) # noqa E501
            verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, nb_verif, verification_embeddings.size(1))) # noqa E501

            enrollment_centroids = enrollment_embeddings.mean(dim=1)
            similarity_matrix = similarity_matrix_centroids(verification_embeddings, enrollment_centroids, 'cpu') # noqa E501

            diff, eer, = 1, 0

            for threshold in [0.01*i+0.5 for i in range(50)]:
                mask = similarity_matrix > threshold

                FAR = (sum([mask[i].float().sum() - mask[i, :, i].float().sum() for i in range(int(hp.test.N))]) / (hp.test.N-1.0)/(float(nb_verif))/hp.test.N) # noqa E501

                FRR = (sum([nb_verif-mask[i,:,i].float().sum() for i in range(int(hp.test.N))]) / (float(nb_verif))/hp.test.N) # noqa E501

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    eer = (FAR+FRR)/2

            batch_avg_eer += eer
            print(f"\nEER : {eer}")

        avg_eer += batch_avg_eer/(batch_id+1)
    avg_EER = avg_eer / hp.test.epochs
    print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))


if __name__ == "__main__":
    test()

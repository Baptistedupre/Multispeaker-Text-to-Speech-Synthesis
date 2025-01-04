import torch
import torch.autograd as grad
import torch.nn.functional as F


def get_centroid(embeddings, j, i, k):
    """
    Args:
        embeddings: shape (n_speakers, n_utterances, embedding_size)
        j: speaker index
        i: utterance index
        k: centroid speaker index
    Returns:
        centroid: shape (embedding_size)
    """
    centroid = 0
    if j != k:
        centroid = torch.mean(embeddings[k, :, :], dim=0)
    else:
        for utterance in range(embeddings.shape[1]):
            if utterance != i:
                centroid += embeddings[j, utterance, :]
        centroid /= embeddings.shape[1] - 1

    return centroid


def similarity_matrix(embeddings, weight, bias, device):
    """
    Args:
        embeddings: shape (n_speakers, n_utterances, embedding_size)
    Returns:
        similarity_matrix: shape (n_speakers, n_utterances, n_speakers)
    """
    n_speakers = embeddings.shape[0]
    n_utterances = embeddings.shape[1]
    similarity_matrix = torch.zeros((n_speakers, n_utterances, n_speakers))

    for j in range(n_speakers):
        for i in range(n_utterances):
            emb = embeddings[j, i, :]
            for k in range(n_speakers):
                centroid = get_centroid(embeddings, j, i, k)
                cossim = F.cosine_similarity(emb, centroid, dim=0) + 1e-6
                similarity_matrix[j, i, k] = cossim

    similarity_matrix = weight * similarity_matrix.to(device) + bias
    return similarity_matrix


def similarity_matrix_centroids(embeddings, centroids, device):
    n_speakers_verif = embeddings.shape[0]
    n_utterances = embeddings.shape[1]
    n_speakers_enrolled = centroids.shape[0]
    similarity_matrix = torch.zeros((n_speakers_verif, n_utterances, n_speakers_enrolled)) # noqa E501

    for j in range(n_speakers_verif):
        for i in range(n_utterances):
            emb = embeddings[j, i, :]
            for k in range(n_speakers_enrolled):
                centroid = centroids[k]
                cossim = F.cosine_similarity(emb, centroid, dim=0) + 1e-6
                similarity_matrix[j, i, k] = cossim

    return similarity_matrix


def calc_loss(similarity_matrix):
    idx = list(range(similarity_matrix.size(0)))
    pos = sim_matrix[idx, :, idx]
    neg = torch.log(torch.sum(torch.exp(sim_matrix), dim=2))
    loss_matrix = neg - pos
    loss = torch.sum(loss_matrix)
    return loss, loss_matrix


def shuffle(tensor, perm):
    return tensor[perm]


def unshuffle(tensor, perm):
    res = torch.zeros_like(tensor)
    for i, p in enumerate(perm):
        res[p] = tensor[i]
    return res


if __name__ == "__main__":
    w = grad.Variable(torch.tensor(1.0))
    b = grad.Variable(torch.tensor(0.0))
    embeddings = torch.tensor([[0,1,0],[0,0,1], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]).to(torch.float).reshape(3,2,3) # noqa E501
    sim_matrix = similarity_matrix(embeddings, w, b, 'cpu')
    loss, per_embedding_loss = calc_loss(sim_matrix)
    mask = sim_matrix > 2e-6
    print(mask[0].float().sum())

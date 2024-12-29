import torch


def get_centroid(embeddings, j, i, k):
    """
    Args:
        embeddings: shape (n_speakers, n_utterances, embedding_size)
        i: speaker index
        j: utterance index
        k: centroid speakepip i index
    Returns:
        centroid: shape (embedding_size)
    """
    centroid = 0
    if j != k:
        centroid = torch.mean(embeddings[j, :, :], dim=0)
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

    for i in range(n_speakers):
        for j in range(n_utterances):
            for k in range(n_speakers):
                centroid = get_centroid(embeddings, i, j, k)
                emb = embeddings[i, j, :]
                cossim = torch.cosine_similarity(emb, centroid, dim=0)
                similarity_matrix[i, j, k] = cossim

    similarity_matrix = weight * similarity_matrix.to(device) + bias
    return similarity_matrix


def shuffle(tensor, perm):
    return tensor[perm]


def unshuffle(tensor, perm):
    res = torch.zeros_like(tensor)
    for i, p in enumerate(perm):
        res[p] = tensor[i]
    return res

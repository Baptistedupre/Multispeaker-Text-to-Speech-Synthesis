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
                centroid += embeddings[utterance, :, :]
        centroid /= embeddings.shape[1] - 1

    return centroid


def similarity_matrix(embeddings, weight, bias):
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
                similarity_matrix[i, j, k] = torch.cosine_similarity(embeddings[i, j, :], centroid, dim=0)

    similarity_matrix = weight * similarity_matrix + bias
    return similarity_matrix


import  tqdm
import  torch
from torch import nn
from torch.utils.data import DataLoader


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def extract_features(dataloader: DataLoader,
                     model: nn.Module,
                     device):

    embeddings = []
    model.eval()
    tbar = tqdm.tqdm(dataloader)

    with torch.no_grad():
        for images_batch in tbar:

            # Forward pass
            source_batch = images_batch.to(device)
            embeddings.append(model.forward(source_batch))

    return torch.cat(embeddings, 0).cpu().numpy()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


import  tqdm
import  torch
from torch import nn
from torch.utils.data import DataLoader

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

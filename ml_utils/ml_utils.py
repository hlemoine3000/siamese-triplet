
import  tqdm
import  torch

def extract_features(images,
                     model,
                     device,
                     batch_size=50):

    embeddings = []
    issame_array = []

    model.eval()

    with torch.no_grad():
        tbar = tqdm.tqdm(data_loader, dynamic_ncols=True)
        for images_batch, issame in tbar:
            # Transfer to GPU

            image_batch = images_batch.to(device, non_blocking=True)

            emb = model.forward(image_batch)

            embeddings.append(emb)
            issame_array.append(deepcopy(issame))

        embeddings = torch.cat(embeddings, 0).cpu().numpy()
        issame_array = torch.cat(issame_array, 0).cpu().numpy()
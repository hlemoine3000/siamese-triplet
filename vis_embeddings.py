
import argparse
import sys
import json
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE

import utils
import models
from dataset_utils import dataset
from utils.plotter import VisdomScatterPlotter


class TSNE_Visualizer:
    def __init__(self,
                 perplexity=30.0):

        self.perplexity = perplexity
        self.is_trained = False
        self.X_embedded = None

    def train(self, embeddings):
        self.X_embedded = TSNE(n_components=2,
                               perplexity=self.perplexity ,
                               early_exaggeration=12.0, learning_rate=200.0, n_iter=20000,
                               n_iter_without_progress=1000, min_grad_norm=1e-7,
                               metric="euclidean", init="random", verbose=1,
                               random_state=None, method='exact', angle=0.2).fit_transform(embeddings)
        self.is_trained = True

    def plot(self,
             labels,
             name='Embeddings Visualisation TSNE',
             env_name='Emb_vis',
             port=8097):

        if not self.is_trained:
            print('Nothing to plot. Run train() first.')
            return 1

        my_plot = VisdomScatterPlotter(env_name=env_name,
                                       port=port)

        labels_plot = labels.copy()
        labels_plot += 1
        my_plot.plot(name,
                     self.X_embedded,
                     labels_plot)

        return 0


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/vis_config.json')
    parser.add_argument('--vggface2', action='store_true')
    parser.add_argument('--lfw', action='store_true')
    parser.add_argument('--cox_video1', action='store_true')
    parser.add_argument('--cox_video2', action='store_true')
    parser.add_argument('--cox_video3', action='store_true')
    parser.add_argument('--cox_video4', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--all_cox', action='store_true')

    return parser.parse_args(argv)


if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])

    print('Feature extractor evaluation.')
    print('CONFIGURATION:\t{}'.format(args.config))
    with open(args.config) as json_config_file:
        config = utils.AttrDict(json.load(json_config_file))

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    nrof_folds = config.dataset.cross_validation.num_fold
    fold_tool = utils.FoldGenerator(nrof_folds,
                                    config.dataset.cross_validation.num_train_folds,
                                    config.dataset.cross_validation.num_val_folds)
    train_folds, val_folds, test_folds = fold_tool.get_fold()

    # Data transform
    data_transform = transforms.Compose([
        transforms.Resize((config.model.image_size, config.model.image_size), interpolation=1),
        transforms.ToTensor()
    ])

    vis_dataset = dataset.DatasetS2V_from_subject(config.dataset.coxs2v.still_dir,
                                                  config.dataset.coxs2v.video_dir,
                                                  config.dataset.coxs2v.video_list,
                                                  config.dataset.coxs2v.subject_list,
                                                  data_transform,
                                                  max_samples_per_subject=config.embeddings_visualisation.max_sample_per_class,
                                                  video_only=config.dataset.coxs2v.video_only)

    vis_dataloader = torch.utils.data.DataLoader(vis_dataset,
                                                 num_workers=2,
                                                 batch_size=20)


    # Load model
    print('Loading from checkpoint {}'.format(config.model.checkpoint_path))
    checkpoint = torch.load(config.model.checkpoint_path)
    embedding_size = checkpoint['embedding_size']

    model = models.load_model(config.model.model_arch,
                              embedding_size=embedding_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    plotter = utils.VisdomPlotter(env_name=config.visdom.environment_name, port=config.visdom.port)

    # Launch embeddings extraction
    embeddings = []
    label_array = []
    textlabels = []

    with torch.no_grad():
        for images_batch, label_batch in vis_dataloader:
            # Transfer to GPU

            image_batch = images_batch.to(device, non_blocking=True)

            emb = model.forward(image_batch)

            embeddings.append(emb)
            label_array.append(deepcopy(label_batch))

        embeddings = torch.cat(embeddings, 0).cpu().numpy()
        label_array = torch.cat(label_array, 0).cpu().numpy()

    X_embedded = TSNE(n_components=2, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=20000,
                 n_iter_without_progress=1000, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=1,
                 random_state=None, method='exact', angle=0.2).fit_transform(embeddings)

    my_plot = VisdomScatterPlotter(env_name=config.visdom.environment_name,
                                   port=config.visdom.port)

    label_array += 1
    my_plot.plot('Embeddings Visualisation TSNE',
                 X_embedded,
                 label_array,
                 legends=vis_dataloader.dataset.classes)

    print('Visualisation completed.')
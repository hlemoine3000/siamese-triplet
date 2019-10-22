
import evaluation
from dataset_utils import dataloaders

def get_evaluator(lfw_paths_dict: dict,
                  data_transform,
                  batch_size: int,
                  preload=False) -> evaluation.Evaluator:

    print('TEST SET lfw:\t{}'.format(lfw_paths_dict['test_dir']))
    test_loader = dataloaders.Get_PairsImageFolderLoader(lfw_paths_dict['test_dir'],
                                      lfw_paths_dict['pairs_file'],
                                      data_transform,
                                      batch_size,
                                      preload=preload)
    return evaluation.Pairs_Evaluator(test_loader, 'lfw')


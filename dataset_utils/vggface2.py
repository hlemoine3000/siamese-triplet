
from dataset_utils import dataloaders
import evaluation

def get_vggface2_trainset(train_dir,
                 data_transform,
                 people_per_batch,
                 images_per_person):


    # Set up train loader
    print('TRAIN SET vggface2:\t{}'.format(train_dir))
    return dataloaders.get_image_folder_loader(train_dir,
                                               data_transform,
                                               people_per_batch,
                                               images_per_person)

def get_vggface2_evaluator(test_dir,
                           pairs_file,
                           data_transform,
                           batch_size,
                           preload=False) -> evaluation.Evaluator:


    print('TEST SET vggface2:\t{}'.format(test_dir))
    test_loader = dataloaders.Get_PairsImageFolderLoader(test_dir,
                                      pairs_file,
                                      data_transform,
                                      batch_size,
                                      preload=preload)

    return evaluation.Pairs_Evaluator(test_loader, 'vggface2')

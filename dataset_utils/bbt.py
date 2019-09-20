import scipy.io
import numpy as np
import tqdm
import cv2

from PIL import Image
import torch
from torch.utils import data

from utils.video import Video_Reader
from dataset_utils.dataset import TrackDataset, ImageFolderTrackDataset
from dataset_utils.sampler import TrackSampler

# original_width = 1920
# original_heignt = 800

annotation_width = 1024
annotation_height = 576


def remove_onedim_2darray(array : np.ndarray, to_int=False):

    result_list = []

    for child in array:
        child_list = []
        for element in child:
            if to_int:
                item = int(element[0])
            else:
                item = element[0]
            child_list.append(item)
        result_list.append(child_list)

    return result_list


def remove_onedim_1darray(array : np.ndarray):

    result_list = []

    for element in array:
        result_list.append(str(element[0]))

    return result_list


def Read_Annotation(annotation_path, image_size):

    img_width, img_height = image_size

    mat = scipy.io.loadmat(annotation_path)
    face_tracks = mat['FaceTracks']
    track_number = face_tracks.shape[1]

    frame_number = np.max(face_tracks[0][-1]['frames'])
    annotation_list = [[] for i in range(frame_number + 1)]

    x_list = remove_onedim_2darray(np.squeeze(face_tracks['x']), to_int=True)
    y_list = remove_onedim_2darray(np.squeeze(face_tracks['y']), to_int=True)
    w_list = remove_onedim_2darray(np.squeeze(face_tracks['w']), to_int=True)
    h_list = remove_onedim_2darray(np.squeeze(face_tracks['h']), to_int=True)

    frames_list = remove_onedim_2darray(np.squeeze(face_tracks['frames']))
    timestamps_list = remove_onedim_2darray(np.squeeze(face_tracks['timestamps']))
    trackerId_list = remove_onedim_2darray(np.squeeze(face_tracks['trackerId']))
    groundTruthIdentity_list = remove_onedim_1darray(np.squeeze(face_tracks['groundTruthIdentity']))

    tbar = tqdm.tqdm(range(track_number))
    for track_idx in tbar:

        x = x_list[track_idx]
        y = y_list[track_idx]
        w = w_list[track_idx]
        h = h_list[track_idx]

        for idx, frame in enumerate(frames_list[track_idx]):

            xmin = max(0, x[idx]) #* width_ratio
            xmax = min(img_width, (x[idx] + w[idx])) # * height_ratio
            ymin = max(0, y[idx]) # * width_ratio
            ymax = min(img_height, (y[idx] + h[idx])) # * height_ratio

            annotation_list[frame].append([xmin,
                                           ymin,
                                           xmax,
                                           ymax,
                                           groundTruthIdentity_list[track_idx],
                                           frame,
                                           track_idx])


    for i in range(23):
        annotation_list.insert(0, [])
    for i in range(5):
        annotation_list.pop(9100)

    return annotation_list


class BBTTrackDataset(data.Dataset):
    def __init__(self, annotation_path, movie_path, max_frame, transform=None):

        self.annotation_path = annotation_path
        self.movie_path = movie_path
        self.transform = transform

        # Create video source instance
        print('Initializing video capture at {}'.format(movie_path))
        video_src = Video_Reader(movie_path)

        _, image = video_src.get_frame()

        img_height, img_width, img_channel = image.shape

        print('Reading annotation at {}'.format(annotation_path))
        Annotation_list = Read_Annotation(annotation_path, (img_width, img_height))

        cropped_image_list = []
        sample_tarkid_list = []

        cooccuring_tracks_list = []
        tracksamplesidxs_dict = {}
        gt_labels_list = []
        classes_to_idx = {}
        idx_to_classes = {}
        gt_idx_list = []
        num_gt_classes = 0

        num_frame = min(len(Annotation_list), max_frame)

        print('Extracting face patches.')

        video_src.reset()
        frame_idx = 0
        image_idx = 0
        tbar = tqdm.tqdm(range(num_frame))
        for j in tbar:

            ret, image = video_src.get_frame()
            if not ret:
                break

            if frame_idx < 0:
                frame_annotations = []
            else:
                frame_annotations = Annotation_list[frame_idx]

            track_list =[]
            for annotation in frame_annotations:

                cropped_image = image[annotation[1]: annotation[3], annotation[0]: annotation[2], :]
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                # cropped_image = np.transpose(cropped_image, (2, 0, 1))
                cropped_image = Image.fromarray(cropped_image)
                cropped_image_list.append(cropped_image)
                sample_tarkid_list.append(annotation[6])

                if annotation[6] not in tracksamplesidxs_dict.keys():
                    tracksamplesidxs_dict[annotation[6]] = [image_idx]
                else:
                    tracksamplesidxs_dict[annotation[6]].append(image_idx)

                if annotation[4] not in classes_to_idx.keys():
                    classes_to_idx[annotation[4]] = num_gt_classes
                    idx_to_classes[num_gt_classes] = annotation[4]
                    gt_idx_list.append(num_gt_classes)
                    num_gt_classes += 1
                gt_labels_list.append(classes_to_idx[annotation[4]])

                track_list.append(annotation[6])

                image_idx += 1

            # Note co-occuring tracks
            if len(frame_annotations) > 1:
                track_list.sort()
                if track_list not in cooccuring_tracks_list:
                    cooccuring_tracks_list.append(track_list)

            frame_idx += 1
        print('')

        self.cropped_image_list = cropped_image_list
        self.cooccuring_tracks_list = cooccuring_tracks_list
        self.tracksamplesidxs_dict = tracksamplesidxs_dict
        self.sample_tarkid_list = sample_tarkid_list

    def __getitem__(self, index):
        sample = self.cropped_image_list[index]
        target = self.sample_tarkid_list[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.cropped_image_list)

def get_trainset(pkl_file,
                 samples_per_class: int,
                 transform=None):

    print('TRAIN SET BBT.')
    dataset = TrackDataset(pkl_file,
                           transform=transform)
    sampler = TrackSampler(dataset,
                           samples_per_class)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             num_workers=8,
                                             batch_sampler=sampler,
                                             pin_memory=True)

    return dataloader

def get_testset(test_dir,
                data_transform,
                batch_size):

    video_dataset = ImageFolderTrackDataset(test_dir, transform=data_transform)
    return data.DataLoader(video_dataset,
                                       num_workers=8,
                                       batch_size=batch_size,
                                       pin_memory=True)
